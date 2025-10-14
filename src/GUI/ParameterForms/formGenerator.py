"""!
@file
Form generator.

This script contains the form generator and dynamic inputWidgets.
"""

from PySide6.QtWidgets import QWidget, QFormLayout, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, QSpacerItem, QSizePolicy
from PySide6.QtCore import Qt, Slot, Signal
from src.GUI.utils import *
from src.GUI.ParameterForms.simpleInputWidgets import checkbox, StaticInput, VectorInput, SimpleRadio, SimpleDropdown, XYZRadio, ElementSelectionWidget
from src.GUI.ParameterForms.InputDescription import *
from src.GUI.ParameterForms.inputWidgetInterfaces import *

class FormGenerator(QWidget):
    """!
    FormGenerator.
    
    Generate an input form for interacting with PyPO.
    """
    closed = Signal()
    def __init__ (self, ElementData, readAction = None, addButtons=True, test=False, okText=None):
        """!
        Constructor. Creates a form widget given a list of InputDescription.
        
        @param ElementData List of InputDescription objects.
        @param readAction Function to be called upon clicking OK button.
        @param addButtons Boolean. Determines weather there will be buttons in the bottom of the form.
        @param test Boolean. If true, buttons can be added without the need to provide a readAction.
        @param okText String. Text used as label for the OK button. Defaults to "add"
        """
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
            
    def setupInputs(self):
        """!
        Calls constructor of inputWidget for each InputDescription depending on its inType.
        """
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
            elif inp.inType == inType.dynamicCheckbox:
                input = DynamicCheckboxWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)
            elif inp.inType == inType.elementSelector:
                input = ElementSelectionWidget(inp)
                self.inputs.append(input)
                self.layout.addRow(input)

    def setupButtons(self):
        """!
        Creates close, clear and read buttons in the bottom of the form.
        """
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

    def closeAction(self):
        """!
        Close the form.
        """
        self.closed.emit()

    def clear(self):
        """!
        Clears all inputs of the form.
        """
        for input in self.inputs:
            input.clear()

    def read(self):
        """!
        Reads the form.
        
        @return A dictionary containing the values read form the form.
        """
        paramDict = {}
        for input in self.inputs:
            paramDict.update(input.read())
        return paramDict

class DynamicDropdownWidget(inputWidgetInterface):
    """!
    Dynamic dropdown.

    Dropdown followed by a dynamic section that changes depending on users selection in the dropdown.
    """
    def __init__ (self, inp):
        """!
        Constructor. Creates the form section.
        
        @param inp InputDescription object received from formData.
        """
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

    def makeChildren(self):
        """!
        Creates the nested forms.
        """
        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(QWidget())
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subDict.values():
            child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            child.setContentsMargins(0,0,0,0)
            self.stackedWidget.addWidget(child)
            self.children.append(child)

    @Slot(int)
    def modeUpdate(self, index):
        """!
        Updates the dynamic widget according to users selection.
        
        @param index Index of the option.
        """
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index)
            self.currentChild = self.children[index-1]

    def clear(self):
        """!
        Clears all inputs of the form.
        """
        self.mode.clear()
        for child in self.children:
            child.clear()

    def read(self):
        """!
        Reads the inputs.
        
        @return A dictionary containing the values read form the inputs.
        """
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
        
        
class DynamicRadioWidget(inputWidgetInterface):
    """!
    Dynamic radio button.

    radio button group followed by a dynamic section that changes depending on users selection in the dropdown.
    """
    def __init__ (self, inp):
        """!
        Constructor. Creates the form section.
        
        @param inp InputDescription object received from formData.
        """
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

    def makeChildren(self):
        """!
        Creates the nested forms.
        """
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

    @Slot(int)
    def modeUpdate(self, index):
        """!
        Updates the dynamic widget according to users selection.
        
        @param index Index of the option.
        """
        if self.hasChildren:
            self.stackedWidget.setCurrentIndex(index+1)
            self.currentChild = self.children[index]

    def clear(self):
        """!
        Clears all inputs of the form.
        """
        self.mode.clear()
        for child in self.children:
            child.clear()

    def read(self):
        """!
        Reads the inputs.
        
        @return A dictionary containing the values read from the inputs.
        """
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
    
class DynamicCheckboxWidget(inputWidgetInterface):
    """!
    Dynamic checkbox button.

    Checkbox followed by a dynamic section that changes depending on users selection in the dropdown.
    """
    def __init__ (self, inp):
        """!
        Constructor. Creates the form section.
        
        @param inp InputDescription object received from formData.
        """
        super().__init__()
        self.inputDescription = inp
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        
        self.layout = QFormLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.hasChildren = self.inputDescription.subDict != None
        
        self.mode = checkbox(InputDescription(inType.checkbox, self.inputDescription.outputName, self.inputDescription.label, toolTip=self.inputDescription.toolTip))
        self.mode.stateChanged.connect(self.modeUpdate)
        

        self.layout.addRow(self.mode)

        if self.hasChildren:
            self.children = []
            self.makeChildren()

        self.setLayout(self.layout)
        self.modeUpdate(-1)

    def isChecked(self):
        return self.mode.box.isChecked()

    def makeChildren(self):
        """!
        Creates the nested forms.
        """
        self.stackedWidget = QStackedWidget()
        placeholder = QWidget()
        placeholder.setFixedSize(0,0)
        self.stackedWidget.addWidget(placeholder)
        self.layout.addRow(self.stackedWidget)
        for childInDesList in self.inputDescription.subDict.values():
            self.child = FormGenerator(childInDesList, addButtons=False, readAction=None)
            self.child.setContentsMargins(0,0,0,0)
            self.stackedWidget.addWidget(self.child)
            self.children.append(self.child)

    @Slot(int)
    def modeUpdate(self, index):
        """!
        Updates the dynamic widget according to users selection.
        
        @param index Index of the option.
        """
        if self.isChecked():
            self.stackedWidget.setCurrentIndex(1)
            self.hasChildren = True
        else:
            self.stackedWidget.setCurrentIndex(0)
            self.hasChildren = False

    def clear(self):
        """!
        Clears all inputs of the form.
        """
        self.mode.clear()
        for child in self.children:
            child.clear()

    def read(self):
        """!
        Reads the inputs.
        
        @return A dictionary containing the values read from the inputs.
        """
        try:
            paramDict = {self.inputDescription.outputName: self.isChecked()}
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")


        if self.hasChildren:
            for input in self.child.findChildren(inputWidgetInterface):
                if input.parent().parent() == self.stackedWidget:
                    paramDict.update(input.read())
        return paramDict
