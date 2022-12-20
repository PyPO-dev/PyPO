from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput

import numpy as np



class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDiscription = inp
        
        self.layout = QFormLayout()
        label = self.makeLabelFromString(self.inputDiscription.label)
        self.mode = QComboBox()
        self.mode.addItems(self.inputDiscription.subdict.keys())
        self.mode.activated.connect(self.modeUpdate)
        self.layout.addRow(label, self.mode)


        self.hasChildren = hasattr(self.inputDiscription, 'subdict')
        if self.hasChildren:
            self.placeHolderParent = QWidget(self)
            self.placeHolderParent.setMaximumSize(0,0)
            self.placeHolderParent.setStyleSheet("background: red")
            self.children = []
            self.makeCildren()
        self.modeUpdate()



        self.setStyleSheet("background: orange")

        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)

    def makeCildren(self):
        for childKey, childInDisList in self.inputDiscription.subdict.items():
            child = self.makeChildform(childKey, childInDisList)
            self.children.append(child)
            self.layout.addRow(child)
            
            


    def modeUpdate(self):
        if self.hasChildren:
            for child in self.children:
                child.setParent(self.placeHolderParent)
            self.layout.addRow(self.children[self.mode.currentIndex()])

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
        modeOut = list(self.formData.keys())[ind]
        paramDict = {self.inputDiscription.outputName: modeOut}
        for par, inps in self.inputs[modeOut].items():
            paramDict [par] = np.array(list([float(inp.text())for inp in inps])) 
        return paramDict

    @staticmethod
    def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())
