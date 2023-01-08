from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QHBoxLayout, QMainWindow, \
                                    QPushButton, QSpacerItem, QSizePolicy, QDialog, \
                                    QAction,QVBoxLayout, QGridLayout
                                        
from PyQt5.QtGui import QFont, QIcon, QCursor 
from PyQt5.QtCore import Qt
from src.GUI.selfClosingDialog import selfClosingDialog
from src.GUI.selfClosingDialog_HoverableBtn import HoverOpenBtn
import sys
sys.path.append('../')
sys.path.append('../../')

class MyButton(QPushButton):
    def __init__(self, s):
        super().__init__(s)

class ElementWidget(QWidget):
    def __init__ (self, element, actions, p=None ):
        super().__init__(parent=p)
        self.transformAction = actions[0]
        self.plotElementAction = actions[1]
        self.RemoveElementAction = actions[2]
        self.element = element
        
        layout = QHBoxLayout()
        label = QLabel(element)

        label.setFixedSize(200,39)

        # layout.setContentsMargins(0,0,0,0)
        self.btn = MyButton("⋮")
        self.btn.clicked.connect(self._openOptionsMenu)
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

        btn1 = QPushButton("Transform")
        btn2 = QPushButton("Remove")
        btn3 = QPushButton("Plot")

        btn1.clicked.connect(self.transform)
        btn2.clicked.connect(self.removeElement)
        btn3.clicked.connect(self.plotElement)

        dlgLayout.addWidget(btn1)
        dlgLayout.addWidget(btn2)
        dlgLayout.addWidget(btn3)

        self.dlg.setLayout(dlgLayout)
        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        posi = self.mapToGlobal(self.btn.pos())
        self.dlg.setGeometry(posi.x(), posi.y() ,100,80)
        self.dlg.show()
        
    def _closeOptionsMenu(self):
        self.dlg.close()

    def transform(self):
        self.transformAction(self.element)
        self._closeOptionsMenu()

    def plotElement(self):
        self.plotElementAction(self.element)
        self._closeOptionsMenu()
    
    def removeElement(self):
        self._closeOptionsMenu()
        removeElementDialog = QDialog()
        layout = QGridLayout()
        okBtn = QPushButton("Ok")
        okBtn.clicked.connect(removeElementDialog.accept)
        cancelBtn = QPushButton("Cancel")
        cancelBtn.clicked.connect(removeElementDialog.reject)
        layout.addWidget(QLabel("Do you want to delete element %s?" %(self.element)), 0,0,1,2)
        layout.addWidget(cancelBtn, 1,0)
        layout.addWidget(okBtn, 1,1)
        removeElementDialog.setLayout(layout)

        if removeElementDialog.exec_():
            self.RemoveElementAction(self.element)
            self.setParent(None)
            

   
class FrameWidget(ElementWidget):
    def __init__ (self, element, actions, p=None ):
        super().__init__(element, actions,parent=p)
    
        


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,400)
    wid = ElementWidget("Parabola_0", window)
    
    window.show()
    app.exec_()