from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QHBoxLayout, QMainWindow, \
                                    QPushButton, QSpacerItem, QSizePolicy, QDialog, \
                                    QAction,QVBoxLayout
                                        
from PyQt5.QtGui import QFont, QIcon, QCursor 
from PyQt5.QtCore import Qt
from src.GUI.selfClosingDialog import selfClosingDialog
from src.GUI.selfClosingDialog_HoverableBtn import HoverOpenBtn
import sys
sys.path.append('../')
sys.path.append('../../')



class ElementWidget(QWidget):
    def __init__ (self, element, actions , p=None ):
        super().__init__(parent=p)
        self.transformAction = actions[0]
        self.plotElementAction = actions[1]
        self.element = element
        
        layout = QHBoxLayout()
        label = QLabel(element)

        label.setFixedSize(200,39)

        # self.setContentsMargins(1,0,0,0)
        # self.btn = HoverOpenBtn("btn",self._openOptionsMenu, self._closeOptionsMenu)
        self.btn = QPushButton("⋮")
        self.btn.clicked.connect(self._openOptionsMenu)
        # self.btn.setIcon(QIcon("src/GUI/Images/dots.png"))
        self.btn.setFixedSize(50,39)
        
        layout.addWidget(label)
        layout.addWidget(self.btn)
        layout.setSpacing(40)

        self.setLayout(layout)
        self.setFixedSize(250,60)

    def _openOptionsMenu(self):
        self.dlg = selfClosingDialog(self._closeOptionsMenu, parent = self)

        dlgLayout = QVBoxLayout()
        dlgLayout.setContentsMargins(0,0,0,0)
        dlgLayout.setSpacing(0)

        btn1 = QPushButton("Transform")
        btn2 = QPushButton("Edit")
        btn3 = QPushButton("Plot")

        btn1.clicked.connect(self.transform)
        btn2.clicked.connect(self._closeOptionsMenu)
        btn3.clicked.connect(self.plotElement)

        dlgLayout.addWidget(btn1)
        dlgLayout.addWidget(btn2)
        dlgLayout.addWidget(btn3)

        self.dlg.setLayout(dlgLayout)
        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        posi = self.mapToGlobal(self.btn.pos())
        self.dlg.setGeometry(posi.x(), posi.y() ,100,100)
        self.dlg.show()
        
    def _closeOptionsMenu(self):
        self.dlg.close()

    def transform(self):
        self.transformAction(self.element)
        self._closeOptionsMenu()

    def plotElement(self):
        self.plotElementAction(self.element)
        self._closeOptionsMenu()

   

    
        


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,400)
    wid = ElementWidget("Parabola_0", window)
    
    window.show()
    app.exec_()