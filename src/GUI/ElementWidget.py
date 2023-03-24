from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QHBoxLayout, QMainWindow, \
                                    QPushButton, QSizePolicy, QDialog, \
                                    QAction,QVBoxLayout, QGridLayout
                                        
from PyQt5.QtGui import QFont, QIcon, QCursor 
from PyQt5.QtCore import Qt
from src.GUI.selfClosingDialog import selfClosingDialog
from src.GUI.selfClosingDialog_HoverableBtn import HoverOpenBtn
from src.GUI.utils import MyButton
import sys
sys.path.append('../')
sys.path.append('../../')



class SymDialog(QDialog):
    def __init__(self, stopSlot, clog, msg=None):
        self.msg = "" if msg is None else msg
        super().__init__()

        self.clog = clog

        layout = QGridLayout()
        abortBtn = QPushButton("Abort")
        abortBtn.clicked.connect(self.reject)
        abortBtn.clicked.connect(stopSlot)
        layout.addWidget(QLabel(self.msg), 0,0)
        layout.addWidget(abortBtn, 1,0)
        self.setLayout(layout)
        layout.addWidget(QLabel(self.msg), 0,0)
        layout.addWidget(abortBtn, 1,0)
        self.setLayout(layout)
        
        #layout = QGridLayout()
        #abortBtn = QPushButton("Abort")
        #abortBtn.clicked.connect(self.reject)
        #layout.addWidget(QLabel(self.msg), 0,0)
        #layout.addWidget(abortBtn, 1,0)
        #self.setLayout(layout)
        #layout.addWidget(QLabel(self.msg), 0,0)
        #layout.addWidget(abortBtn, 1,0)
        #self.setLayout(layout)
    def setThread(self, thread):
        self.thread = thread

    def killThread(self):
        self.thread.exit()
        self.reject()
    def reject(self):
        self.clog.info("ABORTED.")
        super().reject()

class RemoveElementDialog(QDialog):
    def __init__(self, elementName):
        super().__init__()
        layout = QGridLayout()
        okBtn = QPushButton("Ok")
        okBtn.clicked.connect(self.accept)
        cancelBtn = QPushButton("Cancel")
        cancelBtn.clicked.connect(self.reject)
        layout.addWidget(QLabel("Do you want to delete element %s?" %(elementName)), 0,0,1,2)
        layout.addWidget(cancelBtn, 1,0)
        layout.addWidget(okBtn, 1,1)
        self.setLayout(layout)

class ElementWidget(QWidget):
    def __init__ (self, name, plotAction, removeAction, transformAction = None, RMSAction = None, 
                    snapAction = None, copyAction = None, removeFromTree = None, p=None):

        super().__init__(parent=p)
        self.plotAction = plotAction
        self.removeAction_ = removeAction
        self.removeFromTree = removeFromTree

        self.actions = {
            "plot": self.plot,
            "remove": self.remove
        }
        if transformAction:
            self.transformAction = transformAction
            self.actions["transform"] = self.transform
        if RMSAction:
            self.RMSFrameAction = RMSAction
            self.actions["RMS"] = self.RMSFrame

        if snapAction:
            self.snapAction = snapAction
            self.actions["snapshot"] = self.snap

        if copyAction:
            self.copyAction = copyAction
            self.actions["copy"] = self.copy
        

        self.name = name
        self.setupUI()

    def setupUI(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        label = QLabel(self.name)

        label.setFixedSize(200,39)

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

        dlgLayout = QVBoxLayout(self.dlg)
        dlgLayout.setContentsMargins(0,0,0,0)

        for name, action in self.actions.items():
            btn = QPushButton(name.capitalize())
            btn.clicked.connect(action)
            dlgLayout.addWidget(btn)

        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        posi = self.mapToGlobal(self.btn.pos())
        self.dlg.setGeometry(posi.x(), posi.y() ,100,80)
        self.dlg.show()
        
    def _closeOptionsMenu(self):
        self.dlg.close()
    
    def transform(self):
        self._closeOptionsMenu()
        self.transformAction(self.name)

    def plot(self):
        self._closeOptionsMenu()
        self.plotAction(self.name)
        
    def RMSFrame(self):
        self._closeOptionsMenu()
        self.RMSFrameAction(self.name)

    def remove(self):
        self._closeOptionsMenu()        
        if RemoveElementDialog(self.name).exec_():
            self.removeAction_(self.name)
            self.setParent(None)
            if self.removeFromTree:
                self.removeFromTree()
    
    def snap(self):
        self._closeOptionsMenu()        
        self.snapAction(self.name)
    
    def copy(self):
        self._closeOptionsMenu()        
        self.copyAction(self.name)


class ReflectorWidget(ElementWidget):
    def __init__(self, name, removeAction, transformAction, plotAction, snapAction, copyAction, removeFromTree=None, p=None):
        super().__init__(name, plotAction, removeAction, transformAction=transformAction, 
                removeFromTree=removeFromTree, snapAction=snapAction, copyAction=copyAction, p=p)
        print("refl 2")

class GroupWidget(ElementWidget):
    def __init__ (self, name, removeAction, plotAction, transformAction, snapAction, copyAction):
        super().__init__(name, plotAction, removeAction, transformAction=transformAction, snapAction=snapAction, copyAction=copyAction)

class FrameWidget(ElementWidget):
    def __init__ (self, name, removeAction, transformAction, plotAction, RMSAction, snapAction, p=None ):
        print("making frm wid")
        super().__init__(name, plotAction, removeAction, transformAction=transformAction, RMSAction=RMSAction, snapAction=snapAction, p=p)

class FieldsWidget(ElementWidget):
    def __init__ (self, name, removeAction, plotAction, p=None ):
        super().__init__(name, plotAction, removeAction, p=p)

class CurrentWidget(ElementWidget):
   def __init__ (self, name, removeAction, plotAction, p=None ):
        super().__init__(name, plotAction, removeAction, p=p)
        
class SFieldsWidget(ElementWidget):
    def __init__ (self, name, removeAction, plotAction, p=None ):
        super().__init__(name, plotAction, removeAction, p=p)
        

if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,400)
    wid = ElementWidget("Parabola_0", window)
    
    window.show()
    app.exec_()
