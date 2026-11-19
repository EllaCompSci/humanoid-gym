"""Signed-distance re-implementation of head.scad (G1 dome head) for STL export.

Every SCAD module is mirrored as a distance field (negative inside); CSG is
min/max. The stock collar comes from Unitree's head_link.STL (voxelised,
signed EDT). Units mm, head_link frame.
"""
import sys, numpy as np
from scipy import ndimage
sys.path.insert(0, __import__('os').path.dirname(__file__))
from voxmesh import load_stl_mm, voxelize

# ---------------- parameters (copied from head.scad) ----------------
C = (-5.0, 0.0); Z0 = 400.; R = 125.; H = 85.; BELLY_H = 14.; WALL = 3.; BAND_R = 1.8
CUT_Z = 372.; Z_CROWN = 452.; RING_W = 14.; FACE_A = 40.; N_PANELS = 5; RIB_W = 10.
FLANGE_W = 38.; FL_OUT = 10.; FL_IN = 13.; GAP = 0.4; FIT = 0.2
PZ0 = Z0 + 3 + GAP; PZ1 = Z_CROWN - GAP
BOSS_D = 9.; BOSS_V = RIB_W/2 + GAP + 7.1; BOSS_Z = (418., 438.); INSERT_D = 4.; INSERT_L = 6.; SCREW_CLR = 3.4
CAM_FRONT = 95.; CAM_Z = 402.; CAM_HALF = (62.4, 14.8); CAM_RND = 10.; CAM_DEPTH = 26.5
HOUSING_X0 = 63.; BEZ_HALF = (66., 23.25); BEZ_CZ = 397.25; BEZ_RND = 13.
TROUGH_Z = (377.5, 388.); TROUGH_Y = (8., 60.); MANIFOLD_X = (66., CAM_FRONT - 22); TRIPOD_X = CAM_FRONT - 13
CH_A = np.array([69., 0, 382.5]); CH_B = np.array([-5., 0, 384.]); CH_R = 6.5; SLEEVE_R = 10.; BORE_R = 9.
HUB_R = 40.; HUB_Z = (386., Z0); SPOKE_W = 10.
BROW_Y = (26., 54.); BROW_Z = (422., 428.); BROW_RND = 3.; BROW_LEN = 20.; BROW_TRAVEL = 11.; BROW_CLR = 0.3
GUIDE_X = (80., CAM_FRONT - 3); GUIDE_WALL = 2.; TAB_Y = (29., 32.); TAB_X = 4.; TAB_TOP = 433.; TAB_HOLE_D = 1.3
SV_L = 22.8; SV_W = 12.2; SV_H = 22.7; SV_SHAFT_OFF = 5.9; SV_TAB_L = 32.3; SV_TAB_T = 2.5; SV_TAB_Z = 15.9; SV_HOLE_SP = 27.8
SV_SHAFT = (78., 440.); SV_Y0 = 1.5; SV_PLATE = 3.; SV_CLR = 0.3; SV_PILOT = 1.6
RIB_ANGLES = [FACE_A + i*(360 - 2*FACE_A)/N_PANELS for i in range(N_PANELS + 1)]
SV_BODY_X = (SV_SHAFT[0] - (SV_L - SV_SHAFT_OFF), SV_SHAFT[0] + SV_SHAFT_OFF)
SV_CX = SV_SHAFT[0] - (SV_L/2 - SV_SHAFT_OFF)
BIG = 1e3

def rk(k, z): return (R - k) * np.sqrt(max(0., 1 - ((z - Z0)/(H - k))**2))

# ---------------- CSG ----------------
U = lambda *fs: np.minimum.reduce(fs)
I = lambda *fs: np.maximum.reduce(fs)
def D(a, *bs): return np.maximum(a, -U(*bs)) if len(bs) > 1 else np.maximum(a, -bs[0])

class P:  # sample points
    def __init__(s, x, y, z): s.x, s.y, s.z = x, y, z
    def my(s): return P(s.x, -s.y, s.z)

# ---------------- primitives ----------------
def ellipsoid(p, c, r):
    q = [(p.x - c[0])/r[0], (p.y - c[1])/r[1], (p.z - c[2])/r[2]]
    k0 = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2)
    k1 = np.sqrt((q[0]/r[0])**2 + (q[1]/r[1])**2 + (q[2]/r[2])**2)
    return k0*(k0 - 1)/(k1 + 1e-9)

def ell(p, k=0.): return ellipsoid(p, (C[0], C[1], Z0), (R - k, R - k, H - k))
def zslab(p, z0, z1): return np.maximum(z0 - p.z, p.z - z1)

def box(p, lo, hi):
    lo = np.asarray(lo, float); hi = np.asarray(hi, float); c = (lo + hi)/2; h = (hi - lo)/2
    dx = np.abs(p.x - c[0]) - h[0]; dy = np.abs(p.y - c[1]) - h[1]; dz = np.abs(p.z - c[2]) - h[2]
    out = np.sqrt(np.maximum(dx, 0)**2 + np.maximum(dy, 0)**2 + np.maximum(dz, 0)**2)
    return out + np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0)

def cyl_z(p, cx, cy, r, z0, z1, r2=None, sx=1., sy=1.):
    rad = np.sqrt(((p.x - cx)/sx)**2 + ((p.y - cy)/sy)**2) * min(sx, sy)
    rz = r if r2 is None else r + (r2 - r)*np.clip((p.z - z0)/(z1 - z0), 0, 1)
    return np.maximum(rad - rz*min(sx, sy), zslab(p, z0, z1))

def at_c(p): return P(p.x - C[0], p.y - C[1], p.z)

def wedge(p, a1, a2, z0, z1):
    q = at_c(p); a1, a2 = np.radians(a1), np.radians(a2)
    d1 = np.sin(a1)*q.x - np.cos(a1)*q.y          # outside past edge a1 (clockwise side)
    d2 = -np.sin(a2)*q.x + np.cos(a2)*q.y         # outside past edge a2
    return np.maximum(np.maximum(d1, d2), zslab(p, z0, z1))

def rib_frame(p, a):
    q = at_c(p); a = np.radians(a)
    return np.cos(a)*q.x + np.sin(a)*q.y, -np.sin(a)*q.x + np.cos(a)*q.y   # u, v

def rib_slab(p, a, w, z0, z1, r0=0., r1=400.):
    u, v = rib_frame(p, a)
    return box(P(u, v, p.z), (r0, -w/2, z0), (r1, w/2, z1))

def radial_cyl(p, a, v0, z0, u0, u1, d):
    u, v = rib_frame(p, a)
    return np.maximum(np.sqrt((v - v0)**2 + (p.z - z0)**2) - d/2, np.maximum(u0 - u, u - u1))

def rrect_x(p, yh, zh, r, cz, x0, x1):
    qy = np.abs(p.y) - (yh - r); qz = np.abs(p.z - cz) - (zh - r)
    d2 = np.sqrt(np.maximum(qy, 0)**2 + np.maximum(qz, 0)**2) + np.minimum(np.maximum(qy, qz), 0) - r
    return np.maximum(d2, np.maximum(x0 - p.x, p.x - x1))

def capsule(p, a, b, r):
    ba = b - a; px, py, pz = p.x - a[0], p.y - a[1], p.z - a[2]
    t = np.clip((px*ba[0] + py*ba[1] + pz*ba[2])/ba.dot(ba), 0, 1)
    return np.sqrt((px - t*ba[0])**2 + (py - t*ba[1])**2 + (pz - t*ba[2])**2) - r

def stadium_x(p, y0, y1, z0, z1, r, x0, x1):
    sz = (z1 - z0)/(2*r); zc = (z0 + z1)/2
    yy = np.clip(p.y, y0 + r, y1 - r)
    d2 = np.sqrt((p.y - yy)**2 + ((p.z - zc)/sz)**2)*min(1, sz) - r*min(1, sz)
    return np.maximum(d2, np.maximum(x0 - p.x, p.x - x1))

def torus_z(p, cx, cy, cz, Rm, r):
    rad = np.sqrt((p.x - cx)**2 + (p.y - cy)**2)
    return np.sqrt((rad - Rm)**2 + (p.z - cz)**2) - r

# ---------------- head modules ----------------
def outer_solid(p):
    return U(I(ell(p), zslab(p, Z0, Z0 + H + 1)),
             I(ellipsoid(p, (C[0], C[1], Z0), (R, R, BELLY_H)), zslab(p, Z0 - BELLY_H - 1, Z0)))

def shell(p, ko=0., ki=WALL): return D(ell(p, ko), ell(p, ki))

def neck(p):
    return U(cyl_z(p, -4.7, 0, 1, CUT_Z - 8, Z0 - 2, sx=27, sy=29.5),
             cyl_z(p, C[0], C[1], 29, 376, HUB_Z[0], r2=HUB_R),
             cyl_z(p, C[0], C[1], HUB_R, HUB_Z[0], HUB_Z[1]))

def spokes(p, os):
    return U(*[I(rib_slab(p, a, SPOKE_W, 388, Z0, HUB_R - 5, R - RING_W + 4), os) for a in RIB_ANGLES])

def brim_ring(p, os):
    rad = np.sqrt((p.x - C[0])**2 + (p.y - C[1])**2)
    ann = np.maximum(np.maximum(rad - (R + 5), (R - RING_W) - rad), zslab(p, Z0 - BELLY_H - 1, Z0 + 3))
    return U(I(os, ann, zslab(p, Z0 - BELLY_H - 1, Z0 + 3)), torus_z(p, C[0], C[1], Z0, R, BAND_R))

def ribs(p):
    e0, e10, e13 = ell(p), ell(p, FL_OUT), ell(p, FL_IN)
    face = wedge(p, -FACE_A, FACE_A, PZ0 - 1, PZ1 + 1)
    out = []
    for a in RIB_ANGLES:
        out.append(I(rib_slab(p, a, RIB_W, Z0, Z_CROWN + 3), D(e0, e13)))
        out.append(D(I(rib_slab(p, a, FLANGE_W, PZ0, PZ1), D(e10, e13)), face))
    return U(*out)

def crown(p):
    return U(I(shell(p), zslab(p, Z_CROWN, Z0 + H + 1)),
             I(D(ell(p, WALL), ell(p, 12)), zslab(p, Z_CROWN, Z_CROWN + 6)))

def face_shell(p, os):
    return I(U(I(shell(p), wedge(p, -FACE_A, FACE_A, Z0, Z_CROWN + 1)),
               I(os, box(p, (CAM_FRONT - WALL, -200, Z0), (CAM_FRONT, 200, Z0 + H)))),
             p.x - CAM_FRONT)

def camera_housing(p):
    return U(rrect_x(p, BEZ_HALF[0], BEZ_HALF[1], BEZ_RND, BEZ_CZ, HOUSING_X0, CAM_FRONT), capsule(p, CH_A, CH_B, SLEEVE_R))

def camera_cuts(p):
    ty = TROUGH_Y[1] - TROUGH_Y[0]
    return U(rrect_x(p, CAM_HALF[0], CAM_HALF[1], CAM_RND, CAM_Z, CAM_FRONT - CAM_DEPTH, CAM_FRONT + 30),
             box(p, (CAM_FRONT - 23, TROUGH_Y[0], TROUGH_Z[0]), (CAM_FRONT + 27, TROUGH_Y[0] + ty, TROUGH_Z[1])),
             box(p, (CAM_FRONT - 23, -TROUGH_Y[1], TROUGH_Z[0]), (CAM_FRONT + 27, -TROUGH_Y[1] + ty, TROUGH_Z[1])),
             box(p, (MANIFOLD_X[0], -TROUGH_Y[1], TROUGH_Z[0]), (MANIFOLD_X[1] + 1, TROUGH_Y[1], TROUGH_Z[1])),
             cyl_z(p, TRIPOD_X, 0, 3.3, 360, 395), cyl_z(p, TRIPOD_X, 0, 6.2, 360, 382),
             capsule(p, CH_A, CH_B, CH_R), cyl_z(p, C[0], C[1], BORE_R, 300, Z0 + 5))

def _guide_half(p):
    g = stadium_x(p, BROW_Y[0] - BROW_CLR - GUIDE_WALL, BROW_Y[1] + BROW_CLR + GUIDE_WALL,
                  BROW_Z[0] - BROW_CLR - GUIDE_WALL, BROW_Z[1] + BROW_CLR + GUIDE_WALL,
                  BROW_RND + GUIDE_WALL, GUIDE_X[0], GUIDE_X[1] + 1)
    x0 = SV_BODY_X[0] - 8; y0 = SV_Y0 + SV_TAB_Z - SV_PLATE
    plate = I(box(p, (x0, y0, BROW_Z[1]), (CAM_FRONT, y0 + SV_PLATE, BROW_Z[1] + 40)), ell(p, 1))
    return U(g, plate)

def _brow_cut_half(p):
    y0 = SV_Y0 + SV_TAB_Z - SV_PLATE - 1
    cuts = [stadium_x(p, BROW_Y[0] - BROW_CLR, BROW_Y[1] + BROW_CLR, BROW_Z[0] - BROW_CLR, BROW_Z[1] + BROW_CLR,
                      BROW_RND, GUIDE_X[0] - 10, CAM_FRONT + 5),
            box(p, (GUIDE_X[0] - 1, TAB_Y[0] - BROW_CLR, BROW_Z[1]),
                (CAM_FRONT - BROW_LEN + TAB_X + BROW_TRAVEL, TAB_Y[1] + BROW_CLR, BROW_Z[1] + 10)),
            box(p, (SV_BODY_X[0] - SV_CLR, y0, SV_SHAFT[1] - SV_W/2 - SV_CLR),
                (SV_BODY_X[0] + SV_L + SV_CLR, y0 + SV_PLATE + 2, SV_SHAFT[1] + SV_W/2 + SV_CLR))]
    for dx in (-SV_HOLE_SP/2, SV_HOLE_SP/2):
        cx = SV_CX + dx
        cuts.append(np.maximum(np.sqrt((p.x - cx)**2 + (p.z - SV_SHAFT[1])**2) - SV_PILOT/2,
                               np.maximum(y0 - p.y, p.y - (y0 + SV_PLATE + 2))))
    return U(*cuts)

def flange_holes(p):
    out = []
    for i in range(N_PANELS):
        for side in (0, 1):
            a = RIB_ANGLES[i + side]; v = BOSS_V if side == 0 else -BOSS_V
            for z in BOSS_Z:
                out.append(radial_cyl(p, a, v, z, np.sqrt(rk(FL_IN, z)**2 - v*v) - 2,
                                      np.sqrt(rk(FL_OUT, z)**2 - v*v) + 1, SCREW_CLR))
    return U(*out)

def frame(p, collar):
    os = outer_solid(p)
    body = U(collar, neck(p), spokes(p, os), brim_ring(p, os), ribs(p), crown(p), face_shell(p, os),
             camera_housing(p), _guide_half(p), _guide_half(p.my()))
    body = np.maximum(body, p.x - CAM_FRONT)
    return D(body, camera_cuts(p), _brow_cut_half(p), _brow_cut_half(p.my()), flange_holes(p))

def panel(p, i):
    a1, a2 = RIB_ANGLES[i], RIB_ANGLES[i + 1]
    parts = [I(shell(p), wedge(p, a1, a2, PZ0, PZ1))]
    holes = [rib_slab(p, a1, RIB_W + 2*GAP, PZ0 - 5, PZ1 + 5), rib_slab(p, a2, RIB_W + 2*GAP, PZ0 - 5, PZ1 + 5)]
    eF = ell(p, FL_OUT - FIT)
    for side in (0, 1):
        a = a1 if side == 0 else a2; v = BOSS_V if side == 0 else -BOSS_V
        for z in BOSS_Z:
            uo = np.sqrt(rk(FL_OUT, z)**2 - v*v)
            parts.append(D(radial_cyl(p, a, v, z, uo - 6, np.sqrt(rk(WALL, z)**2 - v*v) + 1, BOSS_D), eF))
            holes.append(radial_cyl(p, a, v, z, uo - 6, uo + INSERT_L, INSERT_D))
    return D(U(*parts), *holes)

def brow(p, side=1, travel=0.):
    q = P(p.x - travel, p.y if side > 0 else -p.y, p.z)
    bar = stadium_x(q, BROW_Y[0], BROW_Y[1], BROW_Z[0], BROW_Z[1], BROW_RND, CAM_FRONT - BROW_LEN, CAM_FRONT)
    tab = box(q, (CAM_FRONT - BROW_LEN, TAB_Y[0], BROW_Z[1] - 1), (CAM_FRONT - BROW_LEN + TAB_X, TAB_Y[1], TAB_TOP))
    hx = CAM_FRONT - BROW_LEN + TAB_X/2
    hole = np.maximum(np.sqrt((q.x - hx)**2 + (q.z - (TAB_TOP - 2))**2) - TAB_HOLE_D/2,
                      np.maximum(TAB_Y[0] - 1 - q.y, q.y - (TAB_Y[1] + 1)))
    return D(U(bar, tab), hole)

def d455(p):
    """placeholder D455 body (124 x 29 x 26) seated in the pocket, glass flush with the face"""
    return rrect_x(p, 62.0, 14.5, 9.5, CAM_Z, CAM_FRONT - 26, CAM_FRONT - 0.3)

# ---------------- grid evaluation ----------------
class Grid:
    def __init__(s, lo, hi, res):
        s.res = res; s.lo = np.array(lo, float)
        s.n = tuple(int(np.ceil((h - l)/res)) + 1 for l, h in zip(lo, hi))
    def axes(s): return [s.lo[i] + s.res*np.arange(s.n[i]) for i in range(3)]

def collar_sdf(grid, stl, sub=4):
    """smooth field for the stock collar: occupancy supersampled sub^3 per cell, field = (0.5 - frac)*res"""
    V = load_stl_mm(stl); res = grid.res
    P = V.reshape(-1, 3); m = P[:, 2] < CUT_Z + 5
    lo_i = np.maximum(np.floor((P[m].min(0) - 3 - grid.lo)/res).astype(int), 0)
    hi_i = np.minimum(np.ceil((P[m].max(0) + 3 - grid.lo)/res).astype(int), np.array(grid.n) - 1)
    hi_i[2] = min(hi_i[2], int(np.ceil((CUT_Z + 3 - grid.lo[2])/res)))
    fine = Grid(grid.lo + lo_i*res - res/2 + res/(2*sub), grid.lo + (hi_i + 1)*res - res/2, res/sub)
    occ = voxelize(V, fine).astype(np.float32)
    nb = (np.array(occ.shape)//sub)
    occ = occ[:nb[0]*sub, :nb[1]*sub, :nb[2]*sub].reshape(nb[0], sub, nb[1], sub, nb[2], sub).mean((1, 3, 5))
    out = np.full(grid.n, 0.5*res, np.float32)
    out[lo_i[0]:lo_i[0]+nb[0], lo_i[1]:lo_i[1]+nb[1], lo_i[2]:lo_i[2]+nb[2]] = (0.5 - occ)*res
    zs = grid.axes()[2]
    return np.maximum(out, np.maximum(250 - zs, zs - CUT_Z)[None, None, :]).astype(np.float32)

def evaluate(grid, fields, collar=None, slab=12):
    """fields: dict name -> f(p, collar_slab) ; returns dict of float32 arrays"""
    xs, ys, zs = grid.axes()
    out = {k: np.empty(grid.n, np.float32) for k in fields}
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    for k0 in range(0, len(zs), slab):
        k1 = min(len(zs), k0 + slab); n = k1 - k0
        p = P(np.repeat(X[:, :, None], n, 2), np.repeat(Y[:, :, None], n, 2),
              np.broadcast_to(zs[k0:k1], X.shape + (n,)).copy())
        cs = collar[:, :, k0:k1] if collar is not None else None
        for k, f in fields.items():
            out[k][:, :, k0:k1] = f(p, cs)
    return out
