



from enum import Enum
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QTextEdit, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QPlainTextEdit, QScrollBar

from datetime import datetime



class msgTypes(Enum):
    FormInput_Incorrect = 0,
    RTError = 1
    Calculation_result = 2
    # bla = 3
    # bla = 4
    # bla = 5

class ConsoleGenerator():
    _console = None
    
    @classmethod
    def get(cls):
        if not hasattr(cls, "_console") and cls._console != None:
            # print(cls._console, "$$")
            pass
        else:
            cls._console = Console()
        return cls._console

class Console(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMaximumHeight(300)
        self.setReadOnly(True)

        # self.textCursor().insertText()

    # def log(self, msgType, text):
    #     # msgType = msgTypes.FormInput_Incorrect
    #     now = datetime.now().strftime("[%Y/%m/%d - %H:%M:%S]   ")
    #     logitem = now
    #     if msgType:
    #         logitem += msgType.name
    #     logitem += "\n"
    #     logitem += text
    #     logitem += "\n"
    #     self.appendPlainText(logitem)

