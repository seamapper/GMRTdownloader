# Copyright (c) 2026 Paul Johnson
# SPDX-License-Identifier: MIT

"""Modal progress window for grid downloads and mosaicking."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)


class DownloadProgressDialog(QDialog):
    """Shows download/mosaic progress with a status line and progress bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Progress")
        # Non-modal so the event loop keeps processing QThread signals on all platforms
        # (modal dialogs have caused stalled downloads on macOS).
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        self.status_label = QLabel("Starting...")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

    def begin(self, title: str, message: str = "", maximum: int = 100) -> None:
        """Show the dialog and reset the progress bar."""
        self.setWindowTitle(title)
        self.status_label.setText(message or "Starting...")
        if maximum > 0:
            self.progress_bar.setRange(0, maximum)
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setRange(0, 0)
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def update_progress(
        self,
        value: int | None = None,
        message: str | None = None,
        *,
        maximum: int | None = None,
        indeterminate: bool = False,
    ) -> None:
        """Update bar and/or status text."""
        if maximum is not None:
            if maximum > 0:
                self.progress_bar.setRange(0, maximum)
            else:
                self.progress_bar.setRange(0, 0)
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        if message is not None:
            self.status_label.setText(message)
        if value is not None and self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(min(value, self.progress_bar.maximum()))
        QApplication.processEvents()
