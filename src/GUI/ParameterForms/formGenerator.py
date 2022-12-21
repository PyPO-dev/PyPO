from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
import numpy as np

# from src.GUI.ParameterForms.variableInputWidget import VariableInputWidget
from src.GUI.ParameterForms.SimpleInputWidget import SimpleInput


class FormGenerator(QWidget):
    def __init__ (self, ElementData, readAction = None, addButtons=True):
        super().__init__()
        if addButtons and readAction == None:
            raise Exception("Trying to add buttons with no action provided!")
        self.formData = ElementData
        self.readme = readAction

        self.layout = QFormLayout()

        self.inputs = []
        self.setupInputs()
        if addButtons:
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

    def setupButtons(self):
        addBtn = QPushButton("Add")
        addBtn.clicked.connect(self.readme)
        canselBtn = QPushButton("Cancel")
        canselBtn.clicked.connect(self.cancelAction)
        self.layout.addRow(canselBtn, addBtn)

    def cancelAction(self):
        self.setParent(None)

    def read(self):
        paramDict = {}
        for input in self.inputs:
            paramDict.update(input.read())
        print(paramDict)
        return paramDict
        
    
class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        
        self.layout = QFormLayout()
        self.hasChildren = self.inputDescription.subdict != None

        label = self.makeLabelFromString(self.inputDescription.label)
        self.mode = QComboBox()
        if self.hasChildren:
            self.mode.addItems(self.inputDescription.subdict.keys())
        else:
            self.mode.addItems(self.inputDescription.sublist)
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
        for childKey, childInDesList in self.inputDescription.subdict.items():
            # child = self.makeChildform(childKey, childInDesList)
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            self.childrenn.append(child)
            self.layout.addRow(child)

    def modeUpdate(self):
        if self.hasChildren:
            for child in self.childrenn:
                child.setParent(self.placeHolderParent)
            self.layout.addRow(self.childrenn[self.mode.currentIndex()])
            self.currentChild = self.childrenn[self.mode.currentIndex()]

    # def makeChildform(self, childKey,childIndistList):
    #     childWidget = QWidget()
    #     childLayout = QFormLayout()
    #     childLayout.setContentsMargins(0,0,0,0)
    #     for i in range(len(childIndistList)):
    #         widget = SimpleInput(childIndistList[i])
    #         childLayout.addRow(widget)
    #     childWidget.setLayout(childLayout)
    #     return childWidget
    
    def read(self):
        ind = self.mode.currentIndex()
        if self.hasChildren:
            modeOut = list(self.inputDescription.subdict.keys())[ind]
        else: 
            modeOut = list(self.inputDescription.sublist)[ind]
        paramDict = {self.inputDescription.outputName: modeOut}
        if self.hasChildren:
            for input in self.currentChild.findChildren(SimpleInput):
                paramDict.update(input.read())
        return paramDict

    @staticmethod
    def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())


