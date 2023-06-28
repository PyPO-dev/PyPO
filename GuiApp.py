"""!
@file
Main GUI entry point. Run this file from the terminal and watch the magic happen.
"""

import sys
from PySide6.QtWidgets import QApplication
from src.GUI.MainWindow import PyPOMainWindow

app = QApplication(sys.argv)

win = PyPOMainWindow(parent=None)
win.show()
app.exec_()
