from xml.etree.ElementTree import tostring
from PyQt5 import QtWidgets as qtw
from SystemTab import SystemTab
from PyQt5 import QtCore

# from src.Python.System import System


class MainWindow(qtw.QWidget):
    def __init__ (self):
        super().__init__()
        self.sysNumber = 1
        self.setWindowTitle("POPPy")

        self.SystemTabs = []


        MainVBox = qtw.QVBoxLayout()
        self.newSys_btn = qtw.QPushButton("New System")
        self.newSys_btn.clicked.connect(lambda: self.addTab())
        MainVBox.addWidget(self.newSys_btn)
        elements = ["parabola1", 'Ellipse2',"another element", "and one more"]

        # add tabWidget
        self.tabWidget = qtw.QTabWidget()
        MainVBox.addWidget(self.tabWidget)
        self.addTab()

        self.setLayout(MainVBox)

        self.show()


    def addTab(self):
        tab = SystemTab()
        # self.SysWidget = qtw.QWidget(tab)
        # self.SysWidget.setGeometry(QtCore.QRect(190, 110, 411, 81))

        self.tabWidget.addTab(tab,"System %s" %self.sysNumber)
        self.sysNumber += 1


        # init System
        # s = System()
        # self.Systems.append(s)




app = qtw.QApplication([])
mw = MainWindow()
mw.setStyleSheet("background: #5A0168; color:white")
# mw.resize(800,600)
app.exec_()



