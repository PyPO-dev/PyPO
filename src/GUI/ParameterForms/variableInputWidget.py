from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput

import numpy as np



class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDiscription = inp
        self.placeholderparent = QWidget()
        
        self.labels = []
        self.fields = []

        self.layout = QVBoxLayout()
        self.hasChildren = hasattr(self.inputDiscription, 'subdict')
        # print (self.hasChildren)
        self.inputs = {}
        self.mode = QComboBox()
        self.mode.addItems(self.inputDiscription.subdict.keys())
        self.mode.setParent(self.placeholderparent)

        label = self.makeLabelFromString(self.inputDiscription.label)
        toplayout = QHBoxLayout()
        toplayout.addWidget(label)
        toplayout.addWidget(self.mode)
        grandtop = QWidget()
        topWidget = QWidget(parent=grandtop)
        topWidget.setLayout(toplayout)

        self.layout.addWidget(grandtop)

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
        i=0
        for childKey, childInDisList in self.inputDiscription.subdict.items():
            child = self.makeChildform(childKey, childInDisList)
            self.children.append(list(child.findChildren(QWidget)))
            self.layout.addWidget(child)
            i += 1


    def modeChanged(self):
        if self.hasChildren:
            for childlist in self.children:
                for child in childlist:
                    child.setVisible(False)
                    # print("iv", type(child))
            for i in self.children[self.mode.currentIndex()]:
                i.setVisible(True)
                # print("v", type(child))

    def makeChildform(self, childKey,childIndistList):
        childWidget = QWidget()
        childLayout = QVBoxLayout()
        self.inputs[childKey]={}
        for i in range(len(childIndistList)):
            widget = SimpleInput(childIndistList[i])
            self.inputs[childKey][childIndistList[i].outputName] = widget
            childLayout.addWidget(widget)
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

    # def unpackAndAddToForm(self, wid, form):
    #     children = wid.findChildren(QWidget, '', Qt.FindDirectChildrenOnly)
    #     if len(children) != 2:
    #         raise Exception("Number of children insuccichient!!!")
    #     label,edit = tuple(children) 
    #     label.setParent(None)
    #     edit.setParent(None)
    #     form.addRow(label, edit)