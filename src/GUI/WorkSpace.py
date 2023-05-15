from PySide2.QtWidgets import QWidget, QTabWidget, QScrollArea, QVBoxLayout, QSizePolicy
from PySide2.QtCore import Qt
from src.GUI.ElementWidget import ReflectorWidget, GroupWidget, FrameWidget, CurrentWidget, FieldsWidget, CurrentWidget, SFieldsWidget
from src.GUI.Accordion import Accordion

##
# @file
# Defines the workspace widget 

##
# Generate workspace widget for PyPO Gui.
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
        self.POAccordion = Accordion()
        self.addTab(self.addScroll(self.POAccordion), "PO")
        self.setMaximumSize(300, 1000)

    def addScroll(self, wid):
        scroll = QScrollArea()
        scroll.setWidget(wid)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return scroll

    def addReflector(self, name, removeAction, transformAction, plotAction, snapAction, copyAction):
        self.elementLayout.addWidget(ReflectorWidget(name, removeAction, transformAction, plotAction, snapAction, copyAction))
        self.setCurrentIndex(0)

    def addRayTraceFrames(self, name, removeAction, transformAction, plotAction, calcRMSfromFrame, snapAction):
        self.frameLayout.addWidget(FrameWidget(name, removeAction, transformAction, plotAction, calcRMSfromFrame, snapAction))
        self.setCurrentIndex(2)

    def addFields(self, name, removeAction, plotAction):
        self.POAccordion.POFields.addWidget(FieldsWidget(name, removeAction, plotAction))
        self.setCurrentIndex(3)

    def addSPOFields(self, name, removeAction, plotAction):
        self.POAccordion.SPOFields.addWidget(SFieldsWidget(name, removeAction, plotAction))
        self.setCurrentIndex(3)
        
    def addCurrent(self, name, removeAction, plotAction):
        self.POAccordion.POCurrents.addWidget(CurrentWidget(name, removeAction, plotAction))
        self.setCurrentIndex(3)

    def addGroup(self, name, transformAction, snapAction, plotAction, removeAction, copyAction, infoAction):
        self.groupLayout.addWidget(GroupWidget(name, transformAction, snapAction, plotAction, removeAction, copyAction, infoAction))
        self.setCurrentIndex(1)
