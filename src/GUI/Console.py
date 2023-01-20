



from enum import Enum
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QPlainTextEdit, QScrollBar

from datetime import datetime

class msgTypes(Enum):
    FormInput_Incorrect = 0,
    RTError = 1
    # bla = 2
    # bla = 3
    # bla = 4
    # bla = 5



class Console(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMaximumHeight(300)
        self.setReadOnly(True)
        self.appendPlainText("********** PyPO Console **********")
        self.log(msgTypes.RTError,"Test")

    def log(self, msgType, text):
        # msgType = msgTypes.FormInput_Incorrect
        now = datetime.now().strftime("[%Y/%m/%d - %H:%M:%S]   ")
        logitem = now
        logitem += msgType.name
        logitem += "\n"
        logitem += text
        logitem += "\n"
        self.appendPlainText(logitem)

