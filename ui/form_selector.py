# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Form dropdown control.

On macOS, QComboBox popups are unreliable in PyQt6 (empty list, no clicks, or
popup off-screen). Use a QToolButton with an attached QMenu instead.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMenu,
    QSizePolicy,
    QToolButton,
    QWidget,
)


class FormSelector(QWidget):
    """
    Dropdown for form layouts. API subset matching QComboBox usage in this app.
    """

    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[str] = []
        self._index = -1
        self._use_menu = sys.platform == "darwin"
        self._combo: QComboBox | None = None
        self._tool_button: QToolButton | None = None
        self._menu: QMenu | None = None
        self._action_group: QActionGroup | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._use_menu:
            self._tool_button = QToolButton()
            self._tool_button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextOnly
            )
            self._tool_button.setPopupMode(
                QToolButton.ToolButtonPopupMode.InstantPopup
            )
            self._tool_button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._tool_button.setMinimumHeight(28)
            self._menu = QMenu(self)
            self._tool_button.setMenu(self._menu)
            layout.addWidget(self._tool_button)
        else:
            self._combo = QComboBox()
            self._combo.setEditable(False)
            self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self._combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._combo.currentIndexChanged.connect(self.currentIndexChanged.emit)
            self._combo.currentTextChanged.connect(self.currentTextChanged.emit)
            layout.addWidget(self._combo)

    def _on_mac_action_triggered(self, action: QAction) -> None:
        if self._action_group is None:
            return
        idx = self._action_group.actions().index(action)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def _rebuild_mac_menu(self) -> None:
        if self._menu is None:
            return
        self._menu.clear()
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)
        self._action_group.triggered.connect(self._on_mac_action_triggered)
        for i, label in enumerate(self._items):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(i == self._index)
            self._action_group.addAction(action)
            self._menu.addAction(action)

    def addItems(self, items: list[str]) -> None:
        self._items = list(items)
        if self._combo is not None:
            self._combo.clear()
            self._combo.addItems(self._items)
        else:
            self._rebuild_mac_menu()
        if self._items and self._index < 0:
            self.setCurrentIndex(0)

    def count(self) -> int:
        return len(self._items)

    def currentIndex(self) -> int:
        if self._combo is not None:
            return self._combo.currentIndex()
        return self._index

    def currentText(self) -> str:
        if self._combo is not None:
            return self._combo.currentText()
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return ""

    def setCurrentIndex(self, index: int) -> None:
        if not self._items:
            self._index = index
            return
        index = max(0, min(index, len(self._items) - 1))
        if self._combo is not None:
            self._combo.setCurrentIndex(index)
            return
        if index == self._index:
            self._update_tool_button_label()
            return
        self._index = index
        if self._action_group is not None:
            for i, action in enumerate(self._action_group.actions()):
                action.setChecked(i == self._index)
        self._update_tool_button_label()
        self.currentIndexChanged.emit(self._index)
        self.currentTextChanged.emit(self.currentText())

    def _update_tool_button_label(self) -> None:
        if self._tool_button is None:
            return
        text = self.currentText() or "—"
        self._tool_button.setText(f"{text}  ▾")
