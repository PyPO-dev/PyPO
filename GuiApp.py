import sys
from PyQt5.QtWidgets import QApplication
from src.GUI.MainWindow import PyPOMainWindow



app = QApplication(sys.argv)
win = PyPOMainWindow(parent=None)
win.show()
app.exec_()
