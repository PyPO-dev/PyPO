from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np

from src.GUI.ParameterForms.variableInputWidget import VariableInputWidget
from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput


class FormGenerator(QWidget):
    def __init__ (self, ElementData, readAction):
        super().__init__()
        self.formData = ElementData
        self.readme = readAction

        self.layout = QFormLayout()

        self.inputs = []
        self.setupInputs()
        self.setupButtons()

        self.setLayout(self.layout)

    def setupInputs(self):
        for inp in self.formData:
            if inp.inType.value < 4:
                input = SimpleInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 4:
                pass
            elif inp.inType.value == 5:
                input = VariableInputWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
          

    # def makeInputVariable(self, inp):
    #     ### make label
    #     subform = VariableInputWidget(inp)
    #     self.subForms[inp.outputName] = subform
    #     self.layout.addRow(subform)

    

    def setupButtons(self):
        addBtn = QPushButton("Add")
        addBtn.clicked.connect(self.readme)
        canselBtn = QPushButton("Cancel")
        canselBtn.clicked.connect(self.cancelAction)
        self.layout.addRow(canselBtn, addBtn)

    def cancelAction(self):
        self.setParent(None)

    def read(self):
        paramDict = {"type": "Parabola"}
        for input in self.inputs:
            paramDict.update(input.read())
            print(type(input))
        
    
        


