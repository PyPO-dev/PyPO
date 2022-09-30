from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import QEvent

class selfClosingDialog(QDialog):
    def __init__(self, closeFunc, parent=None):
        super().__init__(parent)
        self.closeFunc = closeFunc

    def leaveEvent(self, event):
        self.closeFunc()