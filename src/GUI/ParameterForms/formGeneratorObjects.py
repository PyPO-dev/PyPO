from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np

from src.GUI.ParameterForms.variableInputWidget import VariableInputWidget
from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput


class FormGeneratorObjects(QWidget):
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
                label, fields = SimpleInput(inp).get()
                self.inputLabels.append(label)
                self.inputFields.append(fields)
                self.layout.addRow(label,fields)
            elif inp.inType.value == 4:
                pass
            elif inp.inType.value == 5:
                labels, fields = VariableInputWidget(inp).get()
                self.inputLabels.append(labels)
                self.inputFields.append(fields)
                for i in range(len(labels)):
                    self.layout.addRow(labels[i],fields[i])


    def makeInputVariable(self, inp):
        ### make label
        subform = VariableInputWidget(inp)
        self.subForms[inp.outputName] = subform
        self.layout.addRow(subform)

    # def makeInputSimple(self,inp):
    #     ### Make inputs and add them to self.inputs
    #     inputs = [QLineEdit() for k in range(inp.numFields)]
    #     self.inputs[inp.outputName] = inputs

    #     ### put them in a widget
    #     inputWidget = QWidget()
    #     inputLayout = QHBoxLayout()
    #     for i in range(inp.numFields):
    #         edit = inputs[i]
    #         if not len(inp.hints) == 0:
    #             edit.setPlaceholderText(str(inp.hints[i]))
    #         inputLayout.addWidget(edit)
    #     inputWidget.setLayout(inputLayout)

    #     ### make label
    #     label = self.makeLabelFromString(inp.label)

    #     ### add to form
    #     self.layout.addRow(label, inputWidget)

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
        # print(self.inputFields)
        for input in self.inputFields:
            # print(input.read())
            paramDict.update(input.read())
        # print(paramDict)

        


