"""Undo the GTK/GLib environment that snap-packaged apps (VS Code snap) leak into child processes.

Symptom when the MuJoCo viewer opens a window:
    GLib-GIO-ERROR: Settings schema 'org.gnome.settings-daemon.plugins.xsettings'
    does not contain a key named 'antialiasing'      -> Aborted (core dumped)
The snap points GLib at its own (older) gsettings schemas and GTK modules.
"""
from __future__ import annotations

import os
import sys

_SNAP_VARS = ("GTK_PATH", "GTK_EXE_PREFIX", "GTK_IM_MODULE_FILE", "GIO_MODULE_DIR", "GSETTINGS_SCHEMA_DIR",
              "LOCPATH", "GDK_PIXBUF_MODULE_FILE", "GDK_PIXBUF_MODULEDIR", "GTK_DATA_PREFIX")
_PATH_VARS = ("XDG_DATA_DIRS", "XDG_CONFIG_DIRS")
_FLAG = "OPENG1_DISPLAY_ENV_FIXED"


def cleaned_env(env=None) -> dict:
    env = dict(os.environ if env is None else env)
    # VS Code's snap keeps the originals as <NAME>_VSCODE_SNAP_ORIG: restore them
    for k in [k for k in env if k.endswith("_VSCODE_SNAP_ORIG")]:
        name, orig = k[: -len("_VSCODE_SNAP_ORIG")], env.pop(k)
        if orig:
            env[name] = orig
        else:
            env.pop(name, None)
    for k in _SNAP_VARS:
        if "/snap/" in env.get(k, ""):
            env.pop(k)
    for k in _PATH_VARS:
        if k in env:
            parts = [p for p in env[k].split(":") if p and "/snap/" not in p]
            if parts:
                env[k] = ":".join(parts)
            else:
                env.pop(k)
    return env


def needs_fix(env=None) -> bool:
    env = os.environ if env is None else env
    return cleaned_env(env) != dict(env)


def fix_snap_env(reexec: bool = False) -> None:
    """Clean the environment. In-process by default (enough if GTK/GLib is not loaded yet);
    with reexec=True the script restarts itself with the clean environment (most robust)."""
    if sys.platform != "linux" or not needs_fix():
        return
    clean = cleaned_env()
    if reexec and not os.environ.get(_FLAG):
        clean[_FLAG] = "1"
        sys.stdout.flush(); sys.stderr.flush()
        os.execve(sys.executable, [sys.executable] + sys.argv, clean)
    for k in list(os.environ):
        if k not in clean:
            del os.environ[k]
    os.environ.update(clean)
