from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QCheckBox, QRadioButton, QButtonGroup, QGridLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QRegExp, Qt, pyqtSlot
from PyQt5.QtGui import QRegExpValidator
from src.GUI.utils import *
from src.GUI.ParameterForms.simpleInputWidgets.simpleInputWidgets import checkbox, StaticInput, VectorInput, SimpleRadio, SimpleDropdown, XYZRadio



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
        # self.layout.setAlignment(self.layout.formAlignment())

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
                input = SimpleDropdown(inp)
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
            elif inp.inType.value == 8:
                input = DynamicDropdownWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType.value == 9:
                input = DynamicRadioWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)

    def setupButtons(self):
        addBtn = QPushButton("Add")
        addBtn.clicked.connect(self.readme)
        canselBtn = QPushButton("Cancel")
        canselBtn.clicked.connect(self.cancelAction)
        self.layout.addRow(canselBtn, addBtn)
        spacerWidget = QWidget()
        spacerLayout = QVBoxLayout(spacerWidget)
        spacerLayout.addItem(QSpacerItem(0,0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.layout.addRow(spacerWidget)

    def cancelAction(self):
        self.setParent(None)

    def read(self):
        paramDict = {}
        for input in self.inputs:
            paramDict.update(input.read())
        # print(paramDict)
        return paramDict
    


    

class DynamicDropdownWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subdict != None
        
        if self.hasChildren:
            options = self.inputDescription.subdict.keys()
        else:
            options = self.inputDescription.options
        
        self.mode = SimpleDropdown(InputDescription(inType.dropdown, self.inputDescription.outputName, self.inputDescription.label, options=options), dynamic = True)
        self.mode.selectionChangedSignal.connect(self.modeUpdate)

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeCildren()

        self.setLayout(self.layout)
        # self.modeUpdate(0)

    def makeCildren(self):
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(QWidget())
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0) ###TODO: is this necessory??
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    @pyqtSlot(int)
    def modeUpdate(self, index):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index)
            self.currentChild = self.children[index-1]

    def read(self):
        print("reading dynamic dropdown")

        # self.modeUpdate()
        ind = self.mode.currentIndex()-1
        if self.hasChildren:
            modeOut = list(self.inputDescription.subdict.keys())[ind]
        else: 
            modeOut = list(self.inputDescription.sublist)[ind]
        paramDict = {self.inputDescription.outputName: modeOut}

        if self.hasChildren:
            for input in self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly):
               paramDict.update(input.read())
        return paramDict

class DynamicRadioWidget(QWidget):
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subdict != None
        
        if self.hasChildren:
            options = list(self.inputDescription.subdict.keys())
        else:
            options = self.inputDescription.options
        
        self.mode = SimpleRadio(InputDescription(inType.dropdown, self.inputDescription.outputName, self.inputDescription.label, options=options))
        self.mode.selectionChangedSignal.connect(self.modeUpdate)

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeCildren()

        self.setLayout(self.layout)
        # self.modeUpdate(0)

    def makeCildren(self):
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(QWidget())
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0) ###TODO: is this necessory??
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    @pyqtSlot(int)
    def modeUpdate(self, index):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index+1)
            self.currentChild = self.children[index]

    def read(self):
        print("reading dynamic radio")
        ind = self.mode.currentIndex()
        if self.hasChildren:
            modeOut = list(self.inputDescription.subdict.keys())[ind]
        else: 
            modeOut = list(self.inputDescription.sublist)[ind]
        paramDict = {self.inputDescription.outputName: modeOut}

        if self.hasChildren:
            for input in self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly):
               paramDict.update(input.read())
        return paramDict



