# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Form dropdown control that works on macOS.

On macOS, QComboBox popups often fail with Fusion + dark palette (invisible list or
no clicks). This widget uses a QPushButton + QMenu on Darwin and QComboBox elsewhere.
"""

from __future__ import annotations

import sys

from PyQt6.QtCore import QPoint, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QSizePolicy,
    QWidget,
)

_MAC_BUTTON_STYLE = """
QPushButton {
    background-color: #454545;
    color: #ffffff;
    border: 1px solid #666666;
    border-radius: 4px;
    padding: 6px 10px;
    text-align: left;
}
QPushButton:hover {
    border-color: #888888;
}
QPushButton:pressed {
    background-color: #3a3a3a;
}
"""

_MAC_MENU_STYLE = """
QMenu {
    background-color: #353535;
    color: #ffffff;
    border: 1px solid #666666;
}
QMenu::item:selected {
    background-color: #2a82da;
}
"""


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

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if self._use_menu:
            self._combo = None
            self._button = QPushButton()
            self._button.setStyleSheet(_MAC_BUTTON_STYLE)
            self._button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._button.setMinimumHeight(28)
            self._button.clicked.connect(self._show_menu)
            layout.addWidget(self._button)
        else:
            self._button = None
            self._combo = QComboBox()
            self._combo.setEditable(False)
            self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self._combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self._combo.currentIndexChanged.connect(self.currentIndexChanged.emit)
            self._combo.currentTextChanged.connect(self.currentTextChanged.emit)
            layout.addWidget(self._combo)

    def addItems(self, items: list[str]) -> None:
        self._items = list(items)
        if self._combo is not None:
            self._combo.clear()
            self._combo.addItems(self._items)
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
            return
        self._index = index
        self._update_button_label()
        self.currentIndexChanged.emit(self._index)
        self.currentTextChanged.emit(self.currentText())

    def _update_button_label(self) -> None:
        if self._button is None:
            return
        text = self.currentText()
        self._button.setText(f"{text}  ▾")

    def _show_menu(self) -> None:
        if not self._items or self._button is None:
            return
        menu = QMenu(self)
        menu.setStyleSheet(_MAC_MENU_STYLE)
        for i, label in enumerate(self._items):
            action = menu.addAction(label)
            if i == self._index:
                font = action.font()
                font.setBold(True)
                action.setFont(font)

            def on_pick(checked: bool = False, idx: int = i) -> None:
                self.setCurrentIndex(idx)

            action.triggered.connect(on_pick)
        pos = self._button.mapToGlobal(QPoint(0, self._button.height()))
        menu.popup(pos)
