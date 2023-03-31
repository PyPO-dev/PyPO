from PyQt5.QtWidgets import QPushButton
from PyQt5.QtCore import QEvent, Qt

class HoverOpenBtn(QPushButton):
    def __init__(self, text, openOtionsFunc, setOpenable, parent=None):
        super().__init__(text, parent)
        self.openFunc = openOtionsFunc
        self.setOpenable = setOpenable

    def enterEvent(self, event):
        self.openFunc()

    # def leaveEvent(self, event):
    #     self.setOpenable()
    