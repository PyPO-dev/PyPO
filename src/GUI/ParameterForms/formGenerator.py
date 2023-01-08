from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton, QStackedWidget, QCheckBox
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator


from numpy import array 

from src.GUI.ParameterForms.InputDescription import *

class MyLabel(QLabel):
    def __init__ (self, s):
        super().__init__(s)
        self.setWordWrap(True)
class MyEdit(QLineEdit):
    def __init__ (self):
        super().__init__()
        self.setAlignment = Qt.AlignTop
        

# Validator_floats = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
# Validator_ints = QRegExpValidator(QRegExp("[-+]?[0-9]*"))
def makeLabelFromString(s):
    return MyLabel(s.replace("_"," ").capitalize())

class FormGenerator(QWidget):
    def __init__ (self, ElementData, readAction = None, addButtons=True, test=False):
        super().__init__()
        if addButtons and readAction == None and not test:
            raise Exception("Trying to add buttons with no action provided!")
        self.formData = ElementData
        self.readme = readAction

        self.layout = QFormLayout()

        self.inputs = []
        self.setupInputs()
        if addButtons:
            self.setupButtons()
        
        self.layout.setContentsMargins(0,0,0,0)

        self.setLayout(self.layout)
    
    def setupInputs(self):
        for inp in self.formData:
            if inp.inType.value == 0:
                input = StaticInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value < 4:
                input = SimpleInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 4:
                input = BooleanInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 5:
                input = DynamicInputWidget(inp)
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
        # print(paramDict)
        for input in self.inputs:
            # print("updating with: ", input.read())
            paramDict.update(input.read())
        return paramDict
    
class StaticInput(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp
        layout = QFormLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        layout.addRow(MyLabel(inp.label), MyLabel(inp.staticValue))
    def read(self):
        return {self.inputDescription.outputName: self.inputDescription.staticValue}


class BooleanInput(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.box = QCheckBox()
        self.label = MyLabel(self.inputDescription.label)
        layout.addWidget(self.label)
        layout.addWidget(self.box)
        self.setLayout(layout)

    def read(self):
        return{self.inputDescription.outputName: self.box.isChecked()}

class SimpleInput(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp

        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setupUI()
        self.setLayout(self.layout)

    def setupUI(self):
        inp = self.inputDescription
        
        self.inputs = [MyEdit() for k in range(inp.numFields)]
        for edit in self.inputs:
            edit.setValidator(None)
        editLayout = QHBoxLayout()
        editLayout.setContentsMargins(2,0,2,0)
        
        for i in range(inp.numFields):
            edit = self.inputs[i]
            edit.setPlaceholderText(str(inp.hints[i]))
            editLayout.addWidget(edit)
        self.editsWid = QWidget()
        self.editsWid.setLayout(editLayout)

        self.label = makeLabelFromString(self.inputDescription.label)
        self.layout.addRow(self.label, self.editsWid)
    
    def read(self):
        l =[] 
        for i in self.inputs:
            # print(self.label.text())
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

class DynamicInputWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
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
            self.children = []
            self.makeCildren()

        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.modeUpdate()

    def makeCildren(self):
        self.stackedWidget = QStackedWidget()
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0) ###TODO: is this necessory??
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    def modeUpdate(self):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(self.mode.currentIndex())
            self.currentChild = self.children[self.mode.currentIndex()]

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


