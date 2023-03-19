from PyQt5.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem, QTabWidget, QPushButton, QScrollArea, QApplication, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QPoint
from src.GUI.ElementWidget import ReflectorWidget
from src.GUI.Acccordion import AccordionSection
from src.GUI.selfClosingDialog import selfClosingDialog
import sys

class Workspace(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


        self.elementSpace = ElementsSpace()
        self.addTab(self.elementSpace, "Elements")
        self.addTab(QWidget(), "Groups")
        self.addTab(QWidget(), "Frames")
        self.addTab(QWidget(), "PO")

        self.setFixedSize(300, 1000)

class ElementsSpace(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)


        
        self.plotRefl = lambda:print("plot"),
        self.removeRefl = lambda x :print(f"removed: {x}")
        self.transformRefl = lambda x :print(f"removed: {x}")
        self.snap = lambda:print("snapped")
        



        
        self.setHeaderLabel("Elements")
        self.refls = QTreeWidgetItem(["Reflectors"], 0)
        self.ffields = QTreeWidgetItem(["Far fields"], 0)
        self.setStyleSheet("font-size: 20px")
        
        self.addTopLevelItems([self.refls, self.ffields])


        self.addReflector("Parabola_0")
        self.addReflector("Parabola_1")
        self.addReflector("Ellipse_0")
        self.addReflector("Ff1_0")
        self.addReflector("faaaar")


        self.refls.setExpanded(True)
        self.ffields.setExpanded(True)





    def addReflector(self, name):
        widget = ReflectorWidget(name, self.removeRefl, self.transformRefl, self.plotRefl, self.snap)
        widget.setAutoFillBackground(True)
        item = QTreeWidgetItem(["cell"])
        widget.backgroundItem = item
        self.refls.addChild(item)
        self.setItemWidget(item, 0, widget)






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