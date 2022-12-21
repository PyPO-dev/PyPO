from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator

from numpy import array 

from src.GUI.ParameterForms.InputDescription import *

# Validator_floats = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
# Validator_ints = QRegExpValidator(QRegExp("[-+]?[0-9]*"))
def makeLabelFromString(s):
        return QLabel(s.replace("_"," ").capitalize())

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

class SimpleInput(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp

        self.layout = QHBoxLayout()
        self.setupUI()
        self.setLayout(self.layout)

    def setupUI(self):
        inp = self.inputDescription
        
        self.inputs = [QLineEdit() for k in range(inp.numFields)]
        for edit in self.inputs:
            edit.setValidator(None)
        editLayout = QHBoxLayout()
        
        for i in range(inp.numFields):
            edit = self.inputs[i]
            edit.setPlaceholderText(str(inp.hints[i]))
            editLayout.addWidget(edit)
        self.editsWid = QWidget()
        self.editsWid.setLayout(editLayout)

        self.label = makeLabelFromString(self.inputDescription.label)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.editsWid)
    
    def read(self):
        l =[] 
        for i in self.inputs:
            print(self.label.text())
            l.append(self.enumToType(self.inputDescription.inType)(i.text()))
        if len(l)>1:        
            if self.inputDescription.oArray:
                l = array(l)
        else:
            l = l[0]
        l = {self.inputDescription.outputName:l}
        return l

    @staticmethod
    def enumToType(intype):
        if intype == inType.integers: return int
        if intype == inType.floats: return float
        if intype == inType.string: return str

    

class VariableInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        
        self.layout = QFormLayout()
        self.hasChildren = self.inputDescription.subdict != None

        label = makeLabelFromString(self.inputDescription.label)
        self.mode = QComboBox()
        if self.hasChildren:
            self.mode.addItems(self.inputDescription.subdict.keys())
        else:
            self.mode.addItems(self.inputDescription.sublist)
        self.mode.activated.connect(self.modeUpdate)
        self.layout.addRow(label, self.mode)

        if self.hasChildren:
            self.childrenn = []
            self.makeCildren()

        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.modeUpdate()

    def makeCildren(self):
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            self.childrenn.append(child)
            self.layout.addRow(child)

    def modeUpdate(self):
        if self.hasChildren:
            for child in self.childrenn:
                child.hide()
            self.childrenn[self.mode.currentIndex()].show()
            self.currentChild = self.childrenn[self.mode.currentIndex()]

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


