"""!
@file 
Defines dialogs for the GUI.
"""

from typing import Optional
from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QLabel
from PySide6.QtCore import Qt, Signal

class SymDialog(QDialog):
    """!
    Dialog to be shown during long calculations. 
    """
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

class RemoveElementDialog(QDialog):
    """!
    Dialog to be shown when trying to delete an element.
    """
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


class selfClosingDialog(QDialog):
    """!
    Frameless dialog containing element options.
    
    This dialog is typically shown at the position of the button that triggers it and closed when the 
    mouse hovers away from it.
    """
    def __init__(self, closeFunc, parent=None):
        super().__init__(parent)
        self.closeFunc = closeFunc

    def leaveEvent(self, event):
        self.closeFunc()


class UnsavedChangesDialog(QDialog):
    """!
    Warning: unsaved changes
    """
    save = Signal()

    def __init__(self, parent, saveAction) -> None:
        super().__init__(parent)
        self.saveAction = saveAction
        layout = QGridLayout()
        saveBtn = QPushButton("Save")
        saveBtn.clicked.connect(self.saveAndClose)
        dontSaveBtn = QPushButton("Don't save")
        dontSaveBtn.clicked.connect(self.accept)
        cancelBtn = QPushButton("Cancel")
        cancelBtn.clicked.connect(self.reject)

        layout.addWidget(QLabel("The current system contains unsaved changed. Do you want to save?"), 0,0,1,3)
        layout.addWidget(cancelBtn, 1,0)
        layout.addWidget(dontSaveBtn, 1,1)
        layout.addWidget(saveBtn, 1,2)
        self.setLayout(layout)

    def saveAndClose(self):
        if self.saveAction():
            self.accept()
        else:
            self.reject()


