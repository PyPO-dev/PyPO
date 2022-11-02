import sys
from PyQt5.QtWidgets import QApplication
from src.GUI.MainWindow import PoppyMainWindow



app = QApplication(sys.argv)
win = PoppyMainWindow(parent=None)
win.show()
app.exec_()
