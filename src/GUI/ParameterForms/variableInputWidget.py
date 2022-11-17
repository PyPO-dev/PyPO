from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp, Qt
import numpy as np



class VariableInputWidget(QWidget):
    def __init__ (self, key, data):
        super().__init__()

        self.key = key
        self.formData = data

        self.layout = QFormLayout()
        
        self.children = []
        
        self.inputs = {}
        self.mode = QComboBox()
        self.mode.addItems(self.formData.keys())

        label = self.makeLabelFromString(self.key)

        self.layout.addRow(label, self.mode)

        for cKey, subdict in self.formData.items():
            child = self.makeChildform(cKey, subdict)
            self.children.append(child)
            self.layout.addRow(child)
            
        self.mode.currentIndexChanged.connect(self.modeChanged)
        self.modeChanged()

        self.setLayout(self.layout)

    def modeChanged(self):
        for w in self.children:
            w.setVisible(False)
        self.children[self.mode.currentIndex()].setVisible(True)

    def makeChildform(self, childKey, childDict):
        childWidget = QWidget()
        childLayout = QFormLayout()
        self.inputs[childKey]={}
        for k,v in childDict.items():
            self.inputs[childKey][k] = self.addInputToLayout(childLayout, k, v)
        childWidget.setLayout(childLayout)
        return childWidget

    def addInputToLayout(self, layout, k, v):
        ### Make inputs and add them to self.inputs
        edits = [QLineEdit() for k in range(v)]

        ### Make inputs only accept numbers
        Validator = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
        for i in range(len(edits)):
            edits[i].setValidator(Validator)

        ### put them in a widget
        inputWidget = QWidget()
        inputLayout = QHBoxLayout()
        for inp in edits:
            inputLayout.addWidget(inp)
        inputWidget.setLayout(inputLayout)

        ### make label
        label = self.makeLabelFromString(k)

        ### add to form
        layout.addRow(label, inputWidget)
        return edits
    
    def read(self):
        ind = self.mode.currentIndex()
        mode = list(self.formData.keys())[ind]
        paramDict = {self.key: mode}
        for par, inps in self.inputs[mode].items():
            paramDict [par] = np.array(list([float(inp.text())for inp in inps])) 
        return paramDict

    @staticmethod
    def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())