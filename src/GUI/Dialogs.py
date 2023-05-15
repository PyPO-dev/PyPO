from PySide2.QtWidgets import QDialog, QGridLayout, QPushButton, QLabel
from PySide2.QtCore import Qt

##
# @file Defines dialogs for the GUI.

##
# Dialog to be shown during long calculations. 
class SymDialog(QDialog):
    def __init__(self, stopSlot, clog, msg=None):
        self.msg = "" if msg is None else msg
        super().__init__()

        self.clog = clog
        self.setWindowFlag(Qt.FramelessWindowHint)


        layout = QGridLayout()
        abortBtn = QPushButton("Abort")
        abortBtn.clicked.connect(self.reject)
        abortBtn.clicked.connect(stopSlot)
        layout.addWidget(QLabel(self.msg), 0,0)
        layout.addWidget(abortBtn, 1,0)
        self.setLayout(layout)
        layout.addWidget(QLabel(self.msg), 0,0)
        layout.addWidget(abortBtn, 1,0)
        self.setLayout(layout)

    def reject(self):
        # self.clog.info("ABORTED.")
        print("ABORTED.")
        super().reject()

##
# Dialog to be shown when trying to delete an element.
class RemoveElementDialog(QDialog):
    def __init__(self, elementName):
        super().__init__()
        layout = QGridLayout()
        okBtn = QPushButton("Ok")
        okBtn.clicked.connect(self.accept)
        cancelBtn = QPushButton("Cancel")
        cancelBtn.clicked.connect(self.reject)
        layout.addWidget(QLabel(f"Do you want to delete element {elementName}?"), 0,0,1,2)
        layout.addWidget(cancelBtn, 1,0)
        layout.addWidget(okBtn, 1,1)
        self.setLayout(layout)


##
# Frameless dialog containing element options.
# 
# This dialog is typically shown at the position of the button that triggers it and closed when the 
# mouse hovers away from it.
class selfClosingDialog(QDialog):
    def __init__(self, closeFunc, parent=None):
        super().__init__(parent)
        self.closeFunc = closeFunc

    def leaveEvent(self, event):
        self.closeFunc()
