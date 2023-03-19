from __future__ import print_function
import sys
from PyQt5.QtWidgets import QApplication
from src.GUI.MainWindow import PyPOMainWindow
from src.GUI.Console import ConsoleGenerator

app = QApplication(sys.argv)


win = PyPOMainWindow(parent=None)
win.show()
app.exec_()
