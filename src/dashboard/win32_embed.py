"""Windows-only helpers for embedding a foreign native window (Open3D's
GLFW-backed window, running in a separate process) as a child panel inside
a PyQt5 widget in a different process.

Deliberately isolated: every function here is safe to call even when
pywin32 is not installed, so src/13_realtime_dashboard.py and
src/dashboard/qt_main_window.py can fall back to the original
free-floating-window behavior (two separate top-level windows) without
crashing. Nothing here touches the existing multiprocessing split between
the Qt event loop (main process) and the Open3D render loop (child
process) -- that architecture is unchanged; this module only changes
where the Open3D window's pixels are displayed on screen.
"""

from __future__ import annotations

try:
    import win32con
    import win32gui
    import win32process

    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


def find_window_by_title(title: str, owner_pid: int | None = None) -> int | None:
    """Find a top-level window's HWND by exact title match, or None.

    FindWindow searches ALL top-level windows system-wide, not just this
    process's own -- if another process (e.g. a second, concurrently
    running instance of this same dashboard) happens to have a window
    with the identical title open at the same time, a bare title search
    can match the wrong window. When owner_pid is given, the match is
    rejected unless the found window actually belongs to that process
    (verified via GetWindowThreadProcessId), and the search keeps looking
    -- this was a real, observed failure mode during development (two
    concurrent sessions each running their own Open3D window titled
    "FOVEAX Phase 9 - 3D Dashboard").
    """
    if not WIN32_AVAILABLE:
        return None
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    if owner_pid is not None:
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        if found_pid != owner_pid:
            return None
    return hwnd


def reparent_as_child(child_hwnd: int, parent_hwnd: int, width: int, height: int) -> None:
    """Reparent child_hwnd under parent_hwnd, strip its title bar/border so
    it reads as an embedded panel, and resize it to fill (width, height).

    Raises RuntimeError (or lets a pywin32 error propagate) on any failure
    -- callers must catch and fall back to the free-floating window; this
    function never silently no-ops.
    """
    if not WIN32_AVAILABLE:
        raise RuntimeError("pywin32 (win32gui/win32con) is not installed.")

    style = win32gui.GetWindowLong(child_hwnd, win32con.GWL_STYLE)
    style &= ~(
        win32con.WS_CAPTION
        | win32con.WS_THICKFRAME
        | win32con.WS_MINIMIZEBOX
        | win32con.WS_MAXIMIZEBOX
        | win32con.WS_SYSMENU
        | win32con.WS_POPUP
        | win32con.WS_BORDER
        | win32con.WS_DLGFRAME
    )
    style |= win32con.WS_CHILD
    win32gui.SetWindowLong(child_hwnd, win32con.GWL_STYLE, style)

    win32gui.SetParent(child_hwnd, parent_hwnd)

    SWP_FRAMECHANGED = 0x0020
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010
    win32gui.SetWindowPos(
        child_hwnd, 0, 0, 0, max(1, width), max(1, height),
        SWP_FRAMECHANGED | SWP_NOZORDER | SWP_NOACTIVATE,
    )


def resize_child(child_hwnd: int, width: int, height: int) -> None:
    """Resize an already-reparented child window to fill (width, height).

    Best-effort: swallows errors since this is called from Qt's
    resizeEvent, where raising would break the host window's own resize.
    """
    if not WIN32_AVAILABLE or not child_hwnd:
        return
    try:
        win32gui.MoveWindow(child_hwnd, 0, 0, max(1, width), max(1, height), True)
    except Exception:
        pass
