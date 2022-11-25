from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np

from src.GUI.ParameterForms.variableInputWidgetcopy import VariableInputWidget



class FormGenerator(QWidget):
    def __init__ (self, ElementData, addAction):
        super().__init__()
        self.formData = ElementData
        self.addElement = addAction

        self.layout = QFormLayout()

        self.inputs = {}
        self.subForms = {}
        self.setupInputs()
        self.setupButtons()

        self.setLayout(self.layout)

    def setupInputs(self):
        for k,v in self.formData.items():
            if type(v) == int:
                self.makeInputSimple(k,v)
            elif type(v) == dict:
                self.makeInputVariable(k,v)

    def makeInputVariable(self, k, v):
        ### make label
        subform = VariableInputWidget(k, v)
        self.subForms[k] = subform
        self.layout.addRow(subform)

    def makeInputSimple(self, k, v):
        ### Make inputs and add them to self.inputs
        inputs = [QLineEdit() for k in range(v)]
        self.inputs[k] = inputs

        ### put them in a widget
        inputWidget = QWidget()
        inputLayout = QHBoxLayout()
        for inp in inputs:
            inputLayout.addWidget(inp)
        inputWidget.setLayout(inputLayout)

        ### make label
        label = self.makeLabelFromString(k)

        ### add to form
        self.layout.addRow(label, inputWidget)

    def setupButtons(self):
        addBtn = QPushButton("Add")
        addBtn.clicked.connect(self.addAction)
        canselBtn = QPushButton("Cancel")
        canselBtn.clicked.connect(self.cancelAction)
        self.layout.addRow(canselBtn, addBtn)

    def cancelAction(self):
        self.setParent(None)

    def addAction(self):
        paramDict = {"type": "Parabola"}
        for k, v in self.formData.items():
            if type(v) == int:
                if v<2:
                    paramDict[k]=self.inputs[k][0].text()
                else:
                    paramDict[k]= np.array([float(le.text()) for le in self.inputs[k]])
            elif type(v) == dict:
                paramDict.update(self.subForms[k].read())
            print(k, paramDict[k])
        self.addElement(paramDict)

    @staticmethod
    def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())





