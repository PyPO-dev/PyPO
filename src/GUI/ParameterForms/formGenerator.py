from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np

from src.GUI.ParameterForms.variableInputWidget import VariableInputWidget
from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput


class FormGenerator(QWidget):
    def __init__ (self, ElementData, addAction):
        super().__init__()
        self.formData = ElementData
        self.addElement = addAction

        self.layout = QFormLayout()

        self.inputLabels = []
        self.inputFields = []
        self.subForms = {}
        self.setupInputs()
        self.setupButtons()

        self.setLayout(self.layout)

    def setupInputs(self):
        for inp in self.formData:
            if inp.inType.value < 4:
                self.layout.addRow(SimpleInput(inp))
            elif inp.inType.value == 4:
                pass
            elif inp.inType.value == 5:
                self.layout.addRow(VariableInputWidget(inp))
          

    # def makeInputVariable(self, inp):
    #     ### make label
    #     subform = VariableInputWidget(inp)
    #     self.subForms[inp.outputName] = subform
    #     self.layout.addRow(subform)

    

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
        for input in self.inputFields:
            paramDict.update(input.read())
    
        


