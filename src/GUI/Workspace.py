from PyQt5.QtWidgets import QWidget, QTreeWidget, QTreeWidgetItem, QTabWidget, QPushButton, QScrollArea, QApplication, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QPoint
from src.GUI.ElementWidget import ReflectorWidget
from src.GUI.Acccordion import AccordionSection
from src.GUI.selfClosingDialog import selfClosingDialog
import sys

class Workspace(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)




        self.actions = {
            "plot": lambda:print("plot"),
            "remove": lambda:print("remove")
        }



        
        self.treeWidget = QTreeWidget()
        self.treeWidget.setHeaderLabel("Elements")
        self.refls = QTreeWidgetItem(["Reflectors"], 0)
        self.ffields = QTreeWidgetItem(["Far fields"], 0)
        self.setStyleSheet("font-size: 20px")
        
        self.treeWidget.addTopLevelItems([self.refls, self.ffields])
        self.treeWidget.itemClicked.connect(self.openItemOptions)

        self.refls.addChild(QTreeWidgetItem(["Parabola_0"]))
        self.refls.addChild(QTreeWidgetItem(["Parabola_1"]))
        self.refls.addChild(QTreeWidgetItem(["Ellipse_0"]))
        self.ffields.addChild(QTreeWidgetItem(["Ff1_0"]))
        self.ffields.addChild(QTreeWidgetItem(["faaaar"]))

        self.refls.setExpanded(True)
        self.ffields.setExpanded(True)


        self.addTab(self.treeWidget, "Elements")
        self.addTab(QWidget(), "Groups")
        self.addTab(QWidget(), "Frames")
        self.addTab(QWidget(), "PO")



        self.setFixedSize(300, 1000)


    def openItemOptions(self, treeItem, col):
        self.selectedItem = treeItem.text(0)
        print(f"{self.selectedItem = }")
        self._openOptionsMenu()
        
    def _openOptionsMenu(self):
        self.dlg = selfClosingDialog(self._closeOptionsMenu, parent = self)

        dlgLayout = QVBoxLayout(self.dlg)
        dlgLayout.setContentsMargins(0,0,0,0)

        for name, action in self.actions.items():
            btn = QPushButton(name.capitalize())
            btn.clicked.connect(action)
            dlgLayout.addWidget(btn)

        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        posi = self.mapToGlobal(QPoint(0,0))
        self.dlg.setGeometry(posi.x(), posi.y() ,100,80)
        self.dlg.show()
        
    def _closeOptionsMenu(self):
        self.dlg.close()