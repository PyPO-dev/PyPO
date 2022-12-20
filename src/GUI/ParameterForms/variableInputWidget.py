from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput

import numpy as np



class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDiscription = inp
        
        self.layout = QFormLayout()
        self.hasChildren = hasattr(self.inputDiscription, 'subdict')

        label = self.makeLabelFromString(self.inputDiscription.label)
        self.mode = QComboBox()
        if self.hasChildren:
            self.mode.addItems(self.inputDiscription.subdict.keys())
        else:
            self.mode.addItems(self.inputDiscription.sublist)
        self.mode.activated.connect(self.modeUpdate)
        self.layout.addRow(label, self.mode)


        if self.hasChildren:
            self.placeHolderParent = QWidget(self)
            self.placeHolderParent.setMaximumSize(0,0)
            self.placeHolderParent.setStyleSheet("background: red")
            self.childrenn = []
            self.makeCildren()
        self.modeUpdate()



        self.setStyleSheet("background: orange")

        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

    def makeCildren(self):
        for childKey, childInDisList in self.inputDiscription.subdict.items():
            child = self.makeChildform(childKey, childInDisList)
            self.childrenn.append(child)
            self.layout.addRow(child)
            
            


    def modeUpdate(self):
        if self.hasChildren:
            for child in self.childrenn:
                child.setParent(self.placeHolderParent)
            self.layout.addRow(self.childrenn[self.mode.currentIndex()])
            self.currentChild = self.childrenn[self.mode.currentIndex()]

    def makeChildform(self, childKey,childIndistList):
        childWidget = QWidget()
        childLayout = QFormLayout()
        childLayout.setContentsMargins(0,0,0,0)
        for i in range(len(childIndistList)):
            widget = SimpleInput(childIndistList[i])
            childLayout.addRow(widget)
        childWidget.setLayout(childLayout)
        return childWidget
    
    def read(self):
        ind = self.mode.currentIndex()
        modeOut = list(self.inputDiscription.subdict.keys())[ind]
        paramDict = {self.inputDiscription.outputName: modeOut}
        if self.hasChildren:
            for input in self.currentChild.findChildren(SimpleInput):
                paramDict.update(input.read())
        return paramDict

    @staticmethod
    def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())
