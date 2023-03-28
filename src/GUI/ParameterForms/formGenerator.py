from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QCheckBox, QRadioButton, QButtonGroup, QGridLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import QRegExp, Qt, pyqtSlot, pyqtSignal
from PyQt5.QtGui import QRegExpValidator
from src.GUI.utils import *
from src.GUI.ParameterForms.simpleInputWidgets.simpleInputWidgets import checkbox, StaticInput, VectorInput, SimpleRadio, SimpleDropdown, XYZRadio, ElementSelectionWidget
from src.GUI.ParameterForms.InputDescription import *


##
# @file
# Form generator.
#
# This script contains the form generator and dynamic inputWidgets.
class FormGenerator(QWidget):
    closed = pyqtSignal()
    ##
    # Constructor. Creates a form widget given a list of InputDescription.
    #
    # @param ElementData List of InputDescription objects.
    # @param readAction Function to be called upon clicking OK button.
    # @param addButtons Boolean. Determines weather there will be buttons in the bottom of the form.
    # @param test Boolean. If true you buttons can be added without the need to provide a readAction.
    # @param okText String. Text used as label for the OK button. Defaults to "add"
    def __init__ (self, ElementData, readAction = None, addButtons=True, test=False, okText="Add"):
        super().__init__()
        if addButtons and readAction == None and not test:
            raise Exception("Trying to add buttons with no action provided!")
        self.formData = ElementData
        self.readme = readAction
        self.okText = okText

        self.layout = QFormLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(0,0,0,0)
        # self.layout.setAlignment(self.layout.formAlignment())

        self.inputs = []
        self.setupInputs()
        if addButtons:
            self.setupButtons()
            
    ##
    # Calls constructor of inputWidget for each InputDescription depending on its inType.
    #
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
            elif inp.inType.value == 10:
                input = ElementSelectionWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)

    ##
    # Creates buttons in the bottom of the form.
    #
    def setupButtons(self):
        addBtn = QPushButton(self.okText)
        addBtn.clicked.connect(self.readme)
        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.closeAction)
        clearBtn = QPushButton("Clear")
        clearBtn.clicked.connect(self.clear)
        btnWidget = QWidget()
        btnlayout = QHBoxLayout(btnWidget)
        btnlayout.addWidget(closeBtn)
        btnlayout.addWidget(clearBtn)
        btnlayout.addWidget(addBtn)
        btnWidget.setContentsMargins(0,4,20,0)
        self.layout.addRow(btnWidget)
        spacerWidget = QWidget()
        spacerLayout = QVBoxLayout(spacerWidget)
        spacerLayout.addItem(QSpacerItem(0,0, QSizePolicy.Expanding, QSizePolicy.MinimumExpanding))
        self.layout.addRow(spacerWidget)

    ##
    # Close the form.
    #
    def closeAction(self):
        self.closed.emit()

    ##
    # Clears all inputs of the form.
    #
    def clear(self):
        for input in self.inputs:
            input.clear()

    ##
    # Reads the form.
    #
    # @return A dectionary containing the values read form the form.
    def read(self):
        paramDict = {}
        for input in self.inputs:
            paramDict.update(input.read())
        return paramDict
    



##
# Dynamic dropdown.
#
# Dropdown followed by a dynamic section that changes depending on users selection in the dropdown
class DynamicDropdownWidget(QWidget):
    ##
    # Constructor. Creates the form section.
    #
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subdict != None
        
        if self.hasChildren:
            options = self.inputDescription.subdict.keys()
        else:
            options = self.inputDescription.options
        
        self.mode = SimpleDropdown(InputDescription(inType.dropdown, self.inputDescription.outputName, self.inputDescription.label, options=options, toolTip= self.inputDescription.toolTip), dynamic = True)
        self.mode.selectionChangedSignal.connect(self.modeUpdate)

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeCildren()

        self.setLayout(self.layout)
        self.modeUpdate(0)

    ##
    # Creates the subforms
    #
    def makeCildren(self):
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(QWidget())
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0) ###TODO: is this necessory??
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    ##
    # Updates the dynamic widget according to users selection.
    #
    # @param index Index of the option
    @pyqtSlot(int)
    def modeUpdate(self, index):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index)
            self.currentChild = self.children[index-1]

    ##
    # Clears all inputs of the form.
    #
    def clear(self):
        self.mode.clear()
        for child in self.children:
            child.clear()

    ##
    # Reads the inputs.
    #
    # @return A dectionary containing the values read form the inputs.
    def read(self):
        try:
            ind = self.mode.currentIndex()-1
            if self.hasChildren:
                modeOut = list(self.inputDescription.subdict.keys())[ind]
            else: 
                modeOut = list(self.inputDescription.sublist)[ind]
            paramDict = {self.inputDescription.outputName: modeOut}
            # print(f"DynamicDropdownWidget {self.stackedWidget.currentIndex() = }")

        except:
            raise Exception(f"Faild to read input: {self.inputDescription.label}")
        
        if self.stackedWidget.currentIndex() == 0:
            raise Exception(f"Faild to read input: {self.inputDescription.label}")
        
        if self.hasChildren:
            for input in self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly):
                paramDict.update(input.read())
        return paramDict
        
        
##
# Dynamic radio button.
#
# radio button group followed by a dynamic section that changes depending on users selection in the dropdown
class DynamicRadioWidget(QWidget):
    ##
    # Constructor. Creates the form section.
    #
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subdict != None
        
        if self.hasChildren:
            options = list(self.inputDescription.subdict.keys())
        else:
            options = self.inputDescription.options
        
        self.mode = SimpleRadio(InputDescription(inType.dropdown, self.inputDescription.outputName, self.inputDescription.label, options=options, toolTip=self.inputDescription.toolTip))
        self.mode.selectionChangedSignal.connect(self.modeUpdate)
        

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeCildren()

        self.setLayout(self.layout)
        self.modeUpdate(-1)

    ##
    # Creates the subforms
    #
    def makeCildren(self):
        self.stackedWidget = QStackedWidget()
        placeholder = QWidget()
        placeholder.setFixedSize(0,0)
        self.stackedWidget.addWidget(placeholder)
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subdict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0) ###TODO: is this necessory??
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    ##
    # Updates the dynamic widget according to users selection.
    #
    # @param index Index of the option
    @pyqtSlot(int)
    def modeUpdate(self, index):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index+1)
            self.currentChild = self.children[index]

    ##
    # Clears all inputs of the form.
    #
    def clear(self):
        self.mode.clear()

    ##
    # Reads the inputs.
    #
    # @return A dectionary containing the values read form the inputs.
    def read(self):
        try:
            ind = self.mode.currentIndex()
            if self.hasChildren:
                modeOut = list(self.inputDescription.subdict.keys())[ind]
            else: 
                modeOut = list(self.inputDescription.sublist)[ind]
            paramDict = {self.inputDescription.outputName: modeOut}
            # print(f"DynamicRadioWidget {self.stackedWidget.currentIndex() = }")
        except:
            raise Exception(f"Faild to read input: {self.inputDescription.label}")
        if self.stackedWidget.currentIndex() == 0:
            raise Exception(f"Faild to read input: {self.inputDescription.label}")

        if self.hasChildren:
            for input in self.currentChild.findChildren(QWidget,options=Qt.FindDirectChildrenOnly):
                paramDict.update(input.read())
        return paramDict


