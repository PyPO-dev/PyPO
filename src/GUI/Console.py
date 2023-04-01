



from enum import Enum
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QTextEdit, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QPlainTextEdit, QScrollBar


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

        # self.setMaximumHeight(300)
        self.setReadOnly(True)

