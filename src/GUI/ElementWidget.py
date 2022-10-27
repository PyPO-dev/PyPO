from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QHBoxLayout, QMainWindow, \
                                    QPushButton, QSpacerItem, QSizePolicy, QDialog, \
                                    QAction
                                        
from PyQt5.QtGui import QFont, QIcon, QCursor 
from PyQt5.QtCore import Qt
from ElementOptionsLayout import ElementOptionsLayout
from selfClosingDialog import selfClosingDialog
from selfClosingDialog_HoverableBtn import HoverOpenBtn
import sys
sys.path.append('../')
sys.path.append('../../')



class ElementWidget(QWidget):
    def __init__ (self, element, p=None ):
        super().__init__(p)
        
        layout = QHBoxLayout()
        label = QLabel(element)

        label.setFixedSize(200,39)

        # self.setContentsMargins(1,0,0,0)
        # self.btn = HoverOpenBtn("btn",self._openOptionsMenu, self._closeOptionsMenu)
        self.btn = QPushButton("btn")
        self.btn.clicked.connect(self._openOptionsMenu)
        self.btn.setIcon(QIcon("src/GUI/Images/dots.png"))
        self.btn.setFixedSize(50,39)
        
        layout.addWidget(label)
        layout.addWidget(self.btn)
        layout.setSpacing(40)

        self.setLayout(layout)
        self.setFixedSize(250,60)

    def _openOptionsMenu(self):
        self.dlg = selfClosingDialog(self._closeOptionsMenu, parent = self)
        
        self.dlg.setLayout(ElementOptionsLayout(self._closeOptionsMenu))
        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        posi = self.mapToGlobal(self.btn.pos())
        self.dlg.setGeometry(posi.x(), posi.y() ,100,100)
        self.dlg.show()
        
    def _closeOptionsMenu(self):
        self.dlg.close()

   

    
        


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,400)
    wid = ElementWidget("Parabola_0", window)
    
    window.show()
    app.exec_()