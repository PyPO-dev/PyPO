from PySide2.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QSpacerItem, QSizePolicy
from PySide2.QtCore import Qt, Slot, Signal
from src.GUI.utils import *
from src.GUI.ParameterForms.simpleInputWidgets import checkbox, StaticInput, VectorInput, SimpleRadio, SimpleDropdown, XYZRadio, ElementSelectionWidget
from src.GUI.ParameterForms.InputDescription import *
from src.GUI.ParameterForms.inputWidgetInterfaces import *


##
# @file
# Form generator.
#
# This script contains the form generator and dynamic inputWidgets.

##
# FormGenerator.
#
# Generate an input form for interacting with PyPO.
class FormGenerator(QWidget):
    closed = Signal()
    ##
    # Constructor. Creates a form widget given a list of InputDescription.
    #
    # @param ElementData List of InputDescription objects.
    # @param readAction Function to be called upon clicking OK button.
    # @param addButtons Boolean. Determines weather there will be buttons in the bottom of the form.
    # @param test Boolean. If true, buttons can be added without the need to provide a readAction.
    # @param okText String. Text used as label for the OK button. Defaults to "add"
    def __init__ (self, ElementData, readAction = None, addButtons=True, test=False, okText=None):
        super().__init__()
        if addButtons and readAction == None and not test:
            raise Exception("Trying to add buttons with no action provided!")
        self.formData = ElementData
        self.readme = readAction
        
        self.okText = okText if okText is not None else "Add"

        self.layout = QFormLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(0,0,0,0)
        self.setFixedWidth(390)
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
            if inp.inType == inType.static:
                input = StaticInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType in [inType.vectorFloats, inType.vectorIntegers, inType.vectorStrings]:
                input = VectorInput(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.checkbox:
                input = checkbox(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.dropdown:
                input = SimpleDropdown(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.radio:
                input = SimpleRadio(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.xyzRadio:
                input = XYZRadio(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.dynamicDropdown:
                input = DynamicDropdownWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.dynamicRadio:
                input = DynamicRadioWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.elementSelector:
                input = ElementSelectionWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)

    ##
    # Creates close, clear and read buttons in the bottom of the form.
    #
    def setupButtons(self):
        addBtn = QPushButton(self.okText)
        closeBtn = QPushButton("Close")
        clearBtn = QPushButton("Clear")

        addBtn.clicked.connect(self.readme)
        closeBtn.clicked.connect(self.closeAction)
        clearBtn.clicked.connect(self.clear)
        
        btnWidget = QWidget()
        btnLayout = QHBoxLayout(btnWidget)
        
        btnLayout.addWidget(closeBtn)
        btnLayout.addWidget(clearBtn)
        btnLayout.addWidget(addBtn)
        
        self.layout.addRow(btnWidget)

        btnWidget.setContentsMargins(0,4,20,10)
        for btn in [addBtn, closeBtn, clearBtn]:
            btn.setFixedSize(110, 30)
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
    # @return A dictionary containing the values read form the form.
    def read(self):
        paramDict = {}
        for input in self.inputs:
            paramDict.update(input.read())
        return paramDict

##
# Dynamic dropdown.
#
# Dropdown followed by a dynamic section that changes depending on users selection in the dropdown.
class DynamicDropdownWidget(inputWidgetInterface):
    ##
    # Constructor. Creates the form section.
    #
    # @param inp InputDescription object received from formData.
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subDict != None
        
        options = self.inputDescription.subDict.keys()
        
        self.mode = SimpleDropdown(InputDescription(inType.dropdown, self.inputDescription.outputName, 
                                                    self.inputDescription.label, options=options, 
                                                    toolTip= self.inputDescription.toolTip), dynamic = True)
        self.mode.selectionChangedSignal.connect(self.modeUpdate)

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeChildren()

        self.setLayout(self.layout)
        self.modeUpdate(0)

    ##
    # Creates the nested forms.
    #
    def makeChildren(self):
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(QWidget())
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subDict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0)
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    ##
    # Updates the dynamic widget according to users selection.
    #
    # @param index Index of the option.
    @Slot(int)
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
    # @return A dictionary containing the values read form the inputs.
    def read(self):
        try:
            ind = self.mode.currentIndex()-1
            if self.hasChildren:
                modeOut = list(self.inputDescription.subDict.keys())[ind]
            else: 
                modeOut = list(self.inputDescription.sublist)[ind]
            paramDict = {self.inputDescription.outputName: modeOut}
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        
        if self.stackedWidget.currentIndex() == 0:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        
        if self.hasChildren:
            for input in self.currentChild.findChildren(inputWidgetInterface, options=Qt.FindChildOption.FindDirectChildrenOnly):
                if input.parent().parent() == self.stackedWidget:
                    paramDict.update(input.read())
        return paramDict
        
        
##
# Dynamic radio button.
#
# radio button group followed by a dynamic section that changes depending on users selection in the dropdown.
class DynamicRadioWidget(inputWidgetInterface):
    ##
    # Constructor. Creates the form section.
    #
    # @param inp InputDescription object received from formData.
    def __init__ (self, inp):
        super().__init__()
        self.inputDescription = inp
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subDict != None
        
        if self.hasChildren:
            options = list(self.inputDescription.subDict.keys())
        else:
            options = self.inputDescription.options
        
        self.mode = SimpleRadio(InputDescription(inType.dropdown, self.inputDescription.outputName, self.inputDescription.label, 
                                                 options=options, toolTip=self.inputDescription.toolTip))
        self.mode.selectionChangedSignal.connect(self.modeUpdate)
        

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeChildren()

        self.setLayout(self.layout)
        self.modeUpdate(-1)

    ##
    # Creates the nested forms.
    #
    def makeChildren(self):
        self.stackedWidget = QStackedWidget()
        placeholder = QWidget()
        placeholder.setFixedSize(0,0)
        self.stackedWidget.addWidget(placeholder)
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subDict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0)
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    ##
    # Updates the dynamic widget according to users selection.
    #
    # @param index Index of the option.
    @Slot(int)
    def modeUpdate(self, index):
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index+1)
            self.currentChild = self.children[index]

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
    # @return A dictionary containing the values read from the inputs.
    def read(self):
        try:
            ind = self.mode.currentIndex()
            if self.hasChildren:
                modeOut = list(self.inputDescription.subDict.keys())[ind]
            else: 
                modeOut = list(self.inputDescription.sublist)[ind]
            paramDict = {self.inputDescription.outputName: modeOut}
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        if self.stackedWidget.currentIndex() == 0:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")

        if self.hasChildren:
            for input in self.currentChild.findChildren(inputWidgetInterface):
                if input.parent().parent() == self.stackedWidget:
                    paramDict.update(input.read())
        return paramDict
