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
                i = SimpleInput(inp)
                self.unpackAndAddToForm(i,self.layout)
            elif inp.inType.value == 4:
                pass
            elif inp.inType.value == 5:
                self.layout.addRow(VariableInputWidget(inp))
                # labels, fields = w.get()
                # self.inputLabels.append(labels)
                # self.inputFields.append(fields)
                # for child in w.findChildren(QWidget, '', Qt.FindDirectChildrenOnly):
                #     for input in child.findChildren(QWidget, '', Qt.FindDirectChildrenOnly):
                #         self.unpackAndAddToForm(input,self.layout)
                #     # self.layout.addRow(labels[i],fields[i])
                # # self.layout.addRow(w)

    def makeInputVariable(self, inp):
        ### make label
        subform = VariableInputWidget(inp)
        self.subForms[inp.outputName] = subform
        self.layout.addRow(subform)

    

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
    

    def unpackAndAddToForm(self, wid, form):
        children = wid.findChildren(QWidget, '', Qt.FindDirectChildrenOnly)
        # print(len(children))
        if len(children) != 2:
            # print(wid.text())
            raise Exception("Number of children insuccichient!!!")
        label,edit = tuple(children) 
        label.setParent(None)
        edit.setParent(None)
        form.addRow(label, edit)

        


