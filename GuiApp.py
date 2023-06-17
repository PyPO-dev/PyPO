# from __future__ import print_function
import sys
from PySide6.QtWidgets import QApplication
from src.GUI.MainWindow import PyPOMainWindow

app = QApplication(sys.argv)

win = PyPOMainWindow(parent=None)
win.show()
app.exec_()
