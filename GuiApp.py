from __future__ import print_function
import sys
from PyQt5.QtWidgets import QApplication
from src.GUI.MainWindow import PyPOMainWindow
from src.GUI.Console import ConsoleGenerator


# import builtins as __builtin__


# def print(*args, **kwargs):
#     __builtin__.print('New print function')
#     return __builtin__.print(*args, **kwargs)

app = QApplication(sys.argv)


win = PyPOMainWindow(parent=None)
win.show()
app.exec_()
