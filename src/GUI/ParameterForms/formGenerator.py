from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton, QStackedWidget, QCheckBox, QRadioButton, QButtonGroup, QGridLayout
from PyQt5.QtCore import QRegExp, Qt
from PyQt5.QtGui import QRegExpValidator
from src.GUI.utils import *
from src.GUI.ParameterForms.simpleInputWidgets.simpleInputWidgets import checkbox, StaticInput, VectorInput, SimpleRadio, XYZRadio



from src.GUI.ParameterForms.InputDescription import *



# Validator_floats = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
# Validator_ints = QRegExpValidator(QRegExp("[-+]?[0-9]*"))


class FormGenerator(QWidget):
    def __init__ (self, ElementData, readAction = None, addButtons=True, test=False):
        super().__init__()
        if addButtons and readAction == None and not test:
            raise Exception("Trying to add buttons with no action provided!")
        self.formData = ElementData
        self.readme = readAction

        self.layout = QFormLayout(self)
        self.layout.setContentsMargins(0,0,0,0)

        self.inputs = []
        self.setupInputs()
        if addButtons:
            self.setupButtons()
            
    def setupInputs(self):
        for inp in self.formData:
            if inp.inType.value == 0:
                input = StaticInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value < 4:
                input = VectorInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 4:
                input = checkbox(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 5:
                input = DynamicInputWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 6:
                input = SimpleRadio(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 7:
                input = XYZRadio(inp)
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
        # print(paramDict)
        return paramDict
    



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
        self.modeUpdate()
        ind = self.mode.currentIndex()
        if self.hasChildren:
            modeOut = list(self.inputDescription.subdict.keys())[ind]
        else: 
            modeOut = list(self.inputDescription.sublist)[ind]
        paramDict = {self.inputDescription.outputName: modeOut}
        if self.hasChildren:
            children = self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly)
            # for c in children:
            #     print(type(c))
            #     try:
            #         print(c.inputDescription.outputName)
            #     except:
            #         pass
        if self.hasChildren:
            for input in self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly):
               paramDict.update(input.read())
        return paramDict



