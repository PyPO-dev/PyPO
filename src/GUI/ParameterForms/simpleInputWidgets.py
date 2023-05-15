from PySide2.QtWidgets import QHBoxLayout, QCheckBox, QFormLayout, QGridLayout, QWidget, QButtonGroup, QRadioButton, QComboBox, QListWidget, QSizePolicy, QLabel
from PySide2.QtCore import Signal
from src.GUI.utils import MyLabel, MyEdit, makeLabelFromString, inType, getValidator
from src.GUI.ParameterForms.InputDescription import InputDescription
from src.GUI.ParameterForms.inputWidgetInterfaces import inputWidgetInterface, selectionWidgetInterface
from numpy import array 

##
# @file provides implementation of form inputs

##
# Exception to be thrown when empty field is encountered.
class EmptyFieldException(Exception):
    pass

##
# Input that cannot be edited by user.
# 
# Can be used to communicate information to user about the form. In this case outputName of the provided InputDescription should be None.
# Can also be used for adding a prefilled value that is visible or invisible to user depending on the 'hidden' value of provided InputDescription (visible by default).
class StaticInput(inputWidgetInterface):
    def __init__ (self, inputDescription:InputDescription):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        if not inputDescription.hidden:
            layout = QFormLayout()
            self.setLayout(layout)
            layout.setContentsMargins(5,4,20,0)
            layout.addRow(MyLabel(inputDescription.label), MyLabel(str(inputDescription.staticValue)))
        
        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)
    
    def clear(self):
        pass

    def read(self):
        try:
            if self.inputDescription.outputName is None:
                return {}
            return {self.inputDescription.outputName: self.inputDescription.staticValue}
        
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        
##
# Implements a checkbox input.
# 
# @param inputDescription.prefill If set to true the checkbox will be checked by default.
class checkbox(inputWidgetInterface):
    def __init__ (self, inputDescription:InputDescription):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))
        layout = QHBoxLayout()
        layout.setContentsMargins(5,4,20,0)
        self.box = QCheckBox()
        self.label = MyLabel(self.inputDescription.label)
        layout.addWidget(self.label)
        layout.addWidget(self.box)
        self.setLayout(layout)

        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)

        if self.inputDescription.prefill:
            self.box.setChecked(True)
    
    def clear(self):
        if self.inputDescription.prefill:
            self.box.setChecked(True)
        else:
            self.box.setChecked(False)

    def read(self):
        try:
            if self.inputDescription.outputName is None:
                return {}
            return{self.inputDescription.outputName: self.box.isChecked()}
        
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        
##
# Implements a single or multiple valued text edit.
# 
# @param inputDescription.numFields Determines the number of fields, default = 1.
# @param inputDescription.hints List of strings to provide hints. The length of this list should match the number of fields. If inputDescription.prefill is True, the hints will be used as prefilled values. 
# @param inputDescription.prefill, If set to true the hints will be used as prefilled values.
class VectorInput(inputWidgetInterface):
    def __init__ (self, inputDescription:InputDescription):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        self.layout = QFormLayout()
        self.layout.setContentsMargins(5,4,20,0)
        self.setupUI()
        self.setLayout(self.layout)

        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)

        if self.inputDescription.prefill:
            self.prefill()

    def setupUI(self):
        inp = self.inputDescription
        
        self.inputs = [MyEdit() for _ in range(inp.numFields)]
        for edit in self.inputs:
            edit.setValidator(getValidator(inp.inType))
        editLayout = QHBoxLayout()
        editLayout.setContentsMargins(5,4,20,0)
        
        for i in range(inp.numFields):
            edit = self.inputs[i]
            if inp.hints:
                edit.setPlaceholderText(str(inp.hints[i]))
            editLayout.addWidget(edit)
        self.editsWid = QWidget()
        self.editsWid.setLayout(editLayout)

        if self.inputDescription.outputName:
            self.label = makeLabelFromString(self.inputDescription.label) 
        self.layout.addRow(self.label, self.editsWid)

    def prefill(self):
        if len(self.inputDescription.hints) != self.inputDescription.numFields:
            raise Exception("Cannot prefill field(s). Number of hints doesn't match number of fields")
        
        for i in range(len(self.inputs)):
            hint = self.inputDescription.hints[i]
            if hint == "":
                raise Exception(f"Empty hint at outputName={self.inputDescription.outputName}")
            if type(hint) != self.enumToType(self.inputDescription.inType):
                raise Exception(f"Cannot prefill. Hint has unexpected type {type(hint)}, expected {self.enumToType(self.inputDescription.inType)}")
            self.inputs[i].setText(str(hint))

    def clear(self):
        for edit in self.inputs:
            edit.setText("")
        if self.inputDescription.prefill:
            self.prefill()

    def read(self):
        try:
            if self.inputDescription.outputName is None:
                return {}
            l =[] 
            for i in range(len(self.inputs)):
                val = self.inputs[i].text()
                ### Uncomment this block to default hints. Not compatible with prefilled values
                # if val == "":
                #     try:
                #         val = str(self.inputDescription.hints[i])
                #     except IndexError:
                #         pass
                if val == "":
                    raise EmptyFieldException(f"Empty field at {self.inputDescription.label}")
                l.append(self.enumToType(self.inputDescription.inType)(val.replace(",",".")))
            if len(l)>1:        
                if self.inputDescription.oArray:
                    l = array(l)
            else:
                l = l[0]
            l = {self.inputDescription.outputName:l}
            return l
        except EmptyFieldException as e:
            raise e
        except Exception as e:
            print(e, type(e))
            raise Exception(f"Failed to read input: {self.inputDescription.label}")

    @staticmethod
    def enumToType(intype):
        if intype == inType.vectorIntegers: return int
        if intype == inType.vectorFloats: return float
        if intype == inType.vectorStrings: return str


##
# Implements a radio button selection widget, used for 'one of many' type of options.
class SimpleRadio(selectionWidgetInterface):

    selectionChangedSignal = Signal(int)

    def __init__ (self, inputDescription:InputDescription):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        layout = QFormLayout(self)
        layout.setContentsMargins(5,4,20,0)

        radioWidget = QWidget()
        radiolayout = QHBoxLayout(radioWidget)

        if self.inputDescription.hints:
            options = self.inputDescription.hints
        else:
            options = self.inputDescription.options
        self.group = QButtonGroup()
        self.group.buttonClicked.connect(self.selectionChanged)
        for i in range(len(options)):
            rb = QRadioButton(options[i])
            self.group.addButton(rb)
            self.group.setId(rb,i)
            radiolayout.addWidget(rb)
        layout.addRow(MyLabel(self.inputDescription.label), radioWidget)

        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)

    
    def read(self):
        try:
            if self.inputDescription.outputName is None:
                return {}
            d = {self.inputDescription.outputName : self.inputDescription.options[self.group.checkedId()]}
            return d
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
    
    def clear(self):
        self.group.setExclusive(False)
        for btn in self.group.buttons():
            btn.setChecked(False)
        self.group.setExclusive(True)
        self.selectionChanged()
    
    def currentIndex(self):
        return self.group.checkedId()

    def selectionChanged(self):
        self.selectionChangedSignal.emit(self.group.checkedId())


##
# Implements a dropdown menu.
# 
# @param inputDescription.options list of strings providing the options for the dropdown.
# @param dynamic Sets "--Select item--" first option. This is used by @see DynamicDropdownWidget to provide a blank form by default.
class SimpleDropdown(selectionWidgetInterface):
    
    selectionChangedSignal = Signal(int)

    def __init__ (self, inputDescription:InputDescription, dynamic = False):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        layout = QFormLayout(self)
        layout.setContentsMargins(5,4,20,0)


        if self.inputDescription.hints:
            options = self.inputDescription.hints
        else:
            options = self.inputDescription.options

        self.comboBox = QComboBox()
        self.comboBox.currentIndexChanged.connect(self.selectionChanged)
        if dynamic:
            self.comboBox.addItem("--Select item--")
        if options:
            self.comboBox.addItems(options)
        
        layout.addRow(MyLabel(self.inputDescription.label), self.comboBox)

        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)


    def read(self):
        try:
            if self.inputDescription.outputName is None:
                return {}
            d = {self.inputDescription.outputName : self.inputDescription.options[self.comboBox.currentIndex()]}
            return d
        
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
    

    def selectionChanged(self):
        self.selectionChangedSignal.emit(self.comboBox.currentIndex())

    def currentIndex(self):
        return self.comboBox.currentIndex()
    
    def clear(self):
        self.comboBox.setCurrentIndex(0)
        self.selectionChanged()

##
# Implements an axes selector.
# 
# Implements a widget with two radio button groups where user can select e.g. x and y of y and z but not x and x.
class XYZRadio(inputWidgetInterface):
    class RadioSubWidget(QWidget):
        def __init__(self, options, name, parent=None):
            super().__init__(parent)
            layout = QGridLayout(self)
            layout.setContentsMargins(5,4,20,0)
            self.buttons = []
            self.name = name

            self.group = QButtonGroup()
            self.group.setExclusive(False)
            self.group.buttonClicked.connect(self.uncheckOthers)
            
            x = 1
            for o in options:
                btn = QRadioButton(o)
                self.group.addButton(btn)            
                self.buttons.append(btn)
                layout.addWidget(btn, 0, x)
                x += 1

        def uncheckOthers(self, caller):
            for btn in self.buttons:
                if btn is not caller:
                    btn.setChecked(False)

        def toggled(self, b):
            btn = self.group.checkedButton()

            if not btn is None:
                self.companion.uncheckOption(btn.text())
            return
        
        def setCompanion(self, c):
            self.companion = c
            for btn in self.buttons:
                btn.toggled.connect(self.toggled)

        def uncheckOption(self, s):
            for btn in self.buttons:
                if btn.text() == s and btn.isChecked(): 
                    btn.toggle()
                    return
                
        def clear(self):
            for btn in self.group.buttons():
                btn.setChecked(False)

        
    def __init__(self, inputDescription, parent=None):
        super().__init__(parent)
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        layout = QFormLayout(self)
        layout.setContentsMargins(5,4,20,0)
        self.r1 = self.RadioSubWidget(["x", "y", "z"],"r1")
        self.r2 = self.RadioSubWidget(["x", "y", "z"],"r2")
        self.r1.setCompanion(self.r2)
        self.r2.setCompanion(self.r1)
        layout.addRow(MyLabel("Abscissa"), self.r1)
        layout.addRow(MyLabel("Ordinate"), self.r2)

        if self.inputDescription.toolTip:
            self.setToolTip(self.inputDescription.toolTip)

    def clear(self):
        self.r1.clear()
        self.r2.clear()

    def read(self):
        if self.r1.group.checkedButton()==None :
            raise EmptyFieldException("Empty field at Abscissa") 
        if self.r2.group.checkedButton()==None:
            raise EmptyFieldException("Empty field at Ordinate") 
        try:
            return {self.inputDescription.outputName:self.r1.group.checkedButton().text() + self.r2.group.checkedButton().text()}
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")
        
##
# Implement a widget for 'many of many' type selection is possible.
# 
# User can select an option from a dropdown and it will appear in a listView. By clicking on an option in the listView it will disappear and return to the dropdown.
# 
# @param inputDescription.option List of options.
class ElementSelectionWidget(inputWidgetInterface):
    def __init__ (self, inputDescription: InputDescription):
        super().__init__()
        self.inputDescription = inputDescription
        self.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding,QSizePolicy.Fixed))

        self.layout = QFormLayout(self)
        elements = self.inputDescription.options

        self.selectedElements = []

        ### make list
        self.selectedList = QListWidget()
        self.selectedList.itemClicked.connect(self.removeItem)

        self.deselectedList = QListWidget()
        self.deselectedList.itemClicked.connect(self.addItem)
        for e in elements:
            self.deselectedList.addItem(e)

        self.layout.addRow(QLabel('Not selected'), QLabel('Selected'))
        self.layout.addRow(self.deselectedList, self.selectedList)

        self.selectedList.setFixedWidth(180)
        self.deselectedList.setFixedWidth(180)
        self.selectedList.setToolTip("Click item to remove")
        self.deselectedList.setToolTip("Click item to add")
        self.setFixedHeight(200)

    def addItem(self, x):
        i = self.deselectedList.takeItem(self.deselectedList.indexFromItem(x).row())
        self.selectedList.addItem(i.text())
        self.selectedElements.append(i.text())

    def removeItem(self, x):
        i = self.selectedList.takeItem(self.selectedList.indexFromItem(x).row())
        self.deselectedList.addItem(i.text())
        self.selectedElements.remove(i.text())

    def clear(self):
        while len(self.selectedElements)>0:
            self.removeItem(self.selectedList.item(0))


    def read(self):
        try:
            return {self.inputDescription.outputName :self.selectedElements}
        except:
            raise Exception(f"Failed to read input: {self.inputDescription.label}")


