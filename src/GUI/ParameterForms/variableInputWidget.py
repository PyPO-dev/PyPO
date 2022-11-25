from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout

from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput

import numpy as np



class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDiscription = inp
        # self.key = key
        # self.formData = data
        self.labels = []
        self.fields = []

        self.layout = QFormLayout()
        self.hasChildren = hasattr(self.inputDiscription, 'subdict')
        print (self.hasChildren)
        self.inputs = {}
        self.mode = QComboBox()
        self.mode.addItems(self.inputDiscription.subdict.keys())

        label = self.makeLabelFromString(self.inputDiscription.label)
        self.labels.append(label)
        self.fields.append(self.mode)

        self.layout.addRow(label, self.mode)

        if self.hasChildren:
            self.children = []
            self.makeCildren()
            
        self.mode.currentIndexChanged.connect(self.modeChanged)
        self.modeChanged()

        self.setLayout(self.layout)

    def get(self):
        if len(self.labels) != len(self.fields):
            raise Exception("Labels and Fields don't match. labels: %d fields: %d" %(len(self.labels), len(self.fields)))
        return (self.labels, self.fields)
        # if self.hasChildren:

    def makeCildren(self):
        print("makeCildren")
        for childKey, childInDisList in self.inputDiscription.subdict.items():
            child = self.makeChildform(childKey, childInDisList)
            self.children.append(child)
            self.layout.addRow(child)


    def modeChanged(self):
        if self.hasChildren:
            for w in self.children:
                w.setVisible(False)
            self.children[self.mode.currentIndex()].setVisible(True)

    def makeChildform(self, childKey,childIndistList):
        childWidget = QWidget()
        childLayout = QFormLayout()
        self.inputs[childKey]={}
        for i in range(len(childIndistList)):
            label, widget = SimpleInput(childIndistList[i]).get()
            self.inputs[childKey][childIndistList[i].outputName] = widget
            self.labels.append(label)
            self.fields.append(widget)
            childLayout.addRow(label, widget)
            # self.inputs[childKey][k] = self.addInputToLayout(childLayout, k, v)
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