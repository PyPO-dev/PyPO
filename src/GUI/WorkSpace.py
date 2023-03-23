from PyQt5.QtWidgets import QWidget, QListWidgetItem, QListWidget, QTreeWidget, QTreeWidgetItem, QTabWidget, QPushButton, QScrollArea, QApplication, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QPoint
from src.GUI.ElementWidget import ReflectorWidget, GroupWidget, FrameWidget, CurrentWidget, FieldsWidget, CurrentWidget, SFieldsWidget
from src.GUI.Acccordion import AccordionSection, Accordion
from src.GUI.selfClosingDialog import selfClosingDialog
from src.GUI.utils import WSSections, MyButton
import sys

class Workspace(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()

    def setupUI(self):
        self.elementSpace = QWidget()
        self.elementLayout = QVBoxLayout(self.elementSpace)
        self.elementLayout.setAlignment(Qt.AlignTop)
        self.addTab(self.addScroll(self.elementSpace), "Elements")
        self.groupSpace = QWidget()
        self.groupLayout = QVBoxLayout(self.groupSpace)
        self.groupLayout.setAlignment(Qt.AlignTop)
        self.addTab(self.addScroll(self.groupSpace), "Groups")
        self.frameSpace = QWidget()
        self.frameLayout = QVBoxLayout(self.frameSpace)
        self.frameLayout.setAlignment(Qt.AlignTop)
        self.addTab(self.addScroll(self.frameSpace), "Frames")
        self.POSpase = QWidget()
        self.POLayout = QVBoxLayout(self.POSpase)
        self.POLayout.setAlignment(Qt.AlignTop)
        self.addTab(self.addScroll(self.POSpase), "PO")
        self.setMaximumSize(300, 1000)

    def addScroll(self, wid):
        scroll = QScrollArea()
        scroll.setWidget(wid)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(300)
        return scroll

    def addReflector(self, name, removeAction, transformAction, plotAction, snapAction):
        print("refl 1")
        self.elementLayout.addWidget(ReflectorWidget(name, removeAction, transformAction, plotAction, snapAction))
        print("refl 3")


    def addRayTraceFrames(self, name, removeAction, transformAction, plotAction, calcRMSfromFrame, snapAction):
        self.frameLayout.addWidget(FrameWidget(name, removeAction, transformAction, plotAction, calcRMSfromFrame, snapAction))

    def addFields(self, name, removeAction, plotAction):
        self.POLayout.addWidget(FieldsWidget(name, removeAction, plotAction))

    def addSPOFields(self, name, removeAction, plotAction):
        self.POLayout.addWidget(SFieldsWidget(name, removeAction, plotAction))

    def addCurrent(self, name, removeAction, plotAction):
        self.POLayout.addWidget(CurrentWidget(name, removeAction, plotAction))

    def addGroup(self, name, transformAction, snapAction, plotAction, removeAction):
        self.groupLayout.addWidget(GroupWidget(name, transformAction, snapAction, plotAction, removeAction))






# class ElementsSpace(QTreeWidget):
    # def __init__(self, parent=None):
    #     super().__init__(parent)


        
    #     self.plotRefl = lambda:print("plot"),
    #     self.removeRefl = lambda x :print(f"removed: {x}")
    #     self.transformRefl = lambda x :print(f"removed: {x}")
    #     self.snap = lambda:print("snapped")

        
    #     self.setHeaderLabel("Elements")
    #     self.refls = QTreeWidgetItem(["Reflectors"], 0)
    #     self.ffields = QTreeWidgetItem(["Far fields"], 0)
    #     self.setStyleSheet("font-size: 20px")
        
    #     self.addTopLevelItems([self.refls, self.ffields])


    #     self.addReflector("Parabola_0")
    #     self.addReflector("Parabola_1")
    #     self.addReflector("Ellipse_0")
    #     self.addReflector("Ff1_0")
    #     self.addReflector("faaaar")


    #     self.refls.setExpanded(True)
    #     self.ffields.setExpanded(True)





    # def addReflector(self, name):
    #     widget = ReflectorWidget(name, self.removeRefl, self.transformRefl, self.plotRefl, self.snap)
    #     widget.setAutoFillBackground(True)
    #     item = QTreeWidgetItem(["cell"])
    #     widget.backgroundItem = item
    #     self.refls.addChild(item)
    #     self.setItemWidget(item, 0, widget)






    # def openItemOptions(self, treeItem, col):
    #     self.selectedItem = treeItem.text(0)
    #     print(f"{self.selectedItem = }")
    #     self._openOptionsMenu()
        
    # def _openOptionsMenu(self):
    #     self.dlg = selfClosingDialog(self._closeOptionsMenu, parent = self)

    #     dlgLayout = QVBoxLayout(self.dlg)
    #     dlgLayout.setContentsMargins(0,0,0,0)

    #     for name, action in self.actions.items():
    #         btn = QPushButton(name.capitalize())
    #         btn.clicked.connect(action)
    #         dlgLayout.addWidget(btn)

    #     self.dlg.setWindowFlag(Qt.FramelessWindowHint)
    #     posi = self.mapToGlobal(QPoint(0,0))
    #     self.dlg.setGeometry(posi.x(), posi.y() ,100,80)
    #     self.dlg.show()
        
    # def _closeOptionsMenu(self):
    #     self.dlg.close()






# class GroupWidget(QWidget):
#     def __init__(self, name, transform, snap, plot, remove, editElements, parent=None) -> None:
#         super().__init__(parent)
#         self.layout = QVBoxLayout(self)
#         self.setMaximumHeight(110)
#         top = QWidget()
#         topLayout = QHBoxLayout(top)

#         self.actions = {
#             "Edit Elements":editElements,  
#             "Transform":    transform,
#             "Snap":         snap,
#             "Plot":         plot,
#             "Remove":       remove
#         }

#         self.options = MyButton("⋮")
#         self.options.clicked.connect(self._openOptionsMenu)
#         self.options.setFixedSize(50,39)
#         label = QLabel(name)
#         label.setFixedHeight(39)

#         topLayout.addWidget(label)
#         topLayout.addWidget(self.options)
#         top.setContentsMargins(0,0,0,0)
#         self.layout.setContentsMargins(0,0,0,0)
#         self.layout.addWidget(top)


#     def _openOptionsMenu(self):
#         self.dlg = selfClosingDialog(self._closeOptionsMenu, parent = self)

#         dlgLayout = QVBoxLayout(self.dlg)
#         dlgLayout.setContentsMargins(0,0,0,0)

#         for name, action in self.actions.items():
#             btn = QPushButton(name.capitalize())
#             btn.clicked.connect(action)
#             dlgLayout.addWidget(btn)

#         self.dlg.setWindowFlag(Qt.FramelessWindowHint)
#         posi = self.mapToGlobal(self.options.pos())
#         self.dlg.setGeometry(posi.x(), posi.y() ,100,80)
#         self.dlg.show()

#     def _closeOptionsMenu(self):
#         self.dlg.close()