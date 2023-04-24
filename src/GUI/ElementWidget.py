from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt
                                        
from src.GUI.Dialogs import selfClosingDialog
from src.GUI.utils import MyButton
from src.GUI.Dialogs import RemoveElementDialog

##
# @file Defines ElementWidget and its subclasses for the workspace.

##
# Defines elementWidgets form the workspace.
# 
# Defines ElementWidgets consisting of a name label and an options button. These element widgets are 
# subclassed into their own type and styled in a distinct color in style.css. The subclasses handle 
# passing the required parameters to the parent. The parent accepts all possible option (actions) that 
# could be given by a subclass. The parent should have a function for each of the possible action.  
class ElementWidget(QWidget):
    def __init__ (self, name, plotAction, removeAction, transformAction = None, RMSAction = None, 
                    snapAction = None, copyAction = None, removeFromTree = None, p=None):

        super().__init__(parent=p)
        self.plotAction = plotAction
        self.removeAction_ = removeAction
        self.removeFromTree = removeFromTree

        self.actions = {
            "Plot": self.plot,
            "Remove": self.remove
        }
        if transformAction:
            self.transformAction = transformAction
            self.actions["Transform"] = self.transform
        if RMSAction:
            self.RMSFrameAction = RMSAction
            self.actions["RMS"] = self.RMSFrame

        if snapAction:
            self.snapAction = snapAction
            self.actions["Snapshot"] = self.snap

        if copyAction:
            self.copyAction = copyAction
            self.actions["Copy"] = self.copy
        

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
        position = self.mapToGlobal(self.btn.pos())
        self.dlg.setGeometry(position.x(), position.y() ,100,80)
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
    def __init__(self, name, removeAction, transformAction, plotAction, snapAction, copyAction, removeFromTree=None, p=None): ##TODO: rename removeFromTree to remove?
        super().__init__(name, plotAction, removeAction, transformAction=transformAction, 
                removeFromTree=removeFromTree, snapAction=snapAction, copyAction=copyAction, p=p)

class GroupWidget(ElementWidget):
    def __init__ (self, name, removeAction, plotAction, transformAction, snapAction, copyAction):
        super().__init__(name, plotAction, removeAction, transformAction=transformAction, snapAction=snapAction, copyAction=copyAction)

class FrameWidget(ElementWidget):
    def __init__ (self, name, removeAction, transformAction, plotAction, RMSAction, snapAction, p=None ):
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
        
