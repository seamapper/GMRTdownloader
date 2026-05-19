# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""
Form dropdown control (QComboBox with app-wide defaults).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QSizePolicy


class FormSelector(QComboBox):
    """Dropdown for form layouts; uses the native Qt style (works on macOS)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(False)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
