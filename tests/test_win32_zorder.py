"""Tests for src/dashboard/win32_embed.py::bring_to_absolute_top -- the fix
for the alert card/threat toast being hidden behind the embedded Open3D
view and per-track native label windows.

Runs against real HWNDs (real Qt widgets forced native via WA_NativeWindow,
the same mechanism qt_main_window.py's alert_container/threat_toast use),
skipped when pywin32/PyQt5 aren't available rather than mocked, since a
mocked win32gui can't prove the actual Windows z-order changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.win32_embed import WIN32_AVAILABLE, bring_to_absolute_top

pytestmark = pytest.mark.skipif(not WIN32_AVAILABLE, reason="pywin32 not installed")


@pytest.fixture
def two_sibling_windows():
    pytest.importorskip("PyQt5")
    import win32gui
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(400, 300)
    parent.show()

    a = QWidget(parent)
    a.setAttribute(Qt.WA_NativeWindow, True)
    a.resize(100, 100)
    a.show()

    b = QWidget(parent)
    b.setAttribute(Qt.WA_NativeWindow, True)
    b.resize(100, 100)
    b.show()

    app.processEvents()
    yield int(a.winId()), int(b.winId()), win32gui
    parent.close()
    app.processEvents()


class TestBringToAbsoluteTop:
    def test_raises_hwnd_above_a_later_sibling(self, two_sibling_windows):
        """b was created after a, so b starts above a in Z-order (matching
        how a per-track label widget created after alert_container would
        bury it). Forcing a to the top must make GetWindow(a, GW_HWNDPREV)
        report no sibling above it."""
        hwnd_a, hwnd_b, win32gui = two_sibling_windows
        GW_HWNDPREV = 3

        bring_to_absolute_top(hwnd_a)
        # No window sits "before" (i.e. above) hwnd_a in Z-order anymore.
        assert win32gui.GetWindow(hwnd_a, GW_HWNDPREV) == 0

    def test_second_call_can_restore_the_other_window_on_top(self, two_sibling_windows):
        hwnd_a, hwnd_b, win32gui = two_sibling_windows
        GW_HWNDPREV = 3

        bring_to_absolute_top(hwnd_a)
        bring_to_absolute_top(hwnd_b)
        assert win32gui.GetWindow(hwnd_b, GW_HWNDPREV) == 0
        # a is no longer topmost once b has been raised after it.
        assert win32gui.GetWindow(hwnd_a, GW_HWNDPREV) != 0

    def test_null_hwnd_is_a_safe_noop(self):
        bring_to_absolute_top(0)
        bring_to_absolute_top(None)

    def test_nonexistent_hwnd_does_not_raise(self):
        bring_to_absolute_top(999_999_999)
