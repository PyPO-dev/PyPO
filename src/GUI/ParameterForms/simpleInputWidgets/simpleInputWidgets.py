from PyQt5.QtWidgets import QHBoxLayout, QCheckBox, QFormLayout, QGridLayout, QWidget, QButtonGroup, QRadioButton, QComboBox
from PyQt5.QtCore import pyqtSignal
from src.GUI.utils import MyLabel, MyEdit, makeLabelFromString, inType
from src.GUI.ParameterForms.InputDescription import InputDescription
from src.GUI.ParameterForms.simpleInputWidgets.inputWidget import inputWidgetInterface, selectionWidgetInterface
from numpy import array 

class StaticInput(inputWidgetInterface):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp
        if not inp.hidden:
            layout = QFormLayout()
            self.setLayout(layout)
            layout.setContentsMargins(0,0,0,0)
            layout.addRow(MyLabel(inp.label), MyLabel(inp.staticValue))

    def read(self):
        if self.inputDescription.outputName is None:
            return {}
        return {self.inputDescription.outputName: self.inputDescription.staticValue}


class checkbox(inputWidgetInterface):
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
        if self.inputDescription.outputName is None:
            return {}
        return{self.inputDescription.outputName: self.box.isChecked()}


class VectorInput(inputWidgetInterface):
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

        self.label = MyLabel("Unnamed")
        if self.inputDescription.outputName:
            self.label = makeLabelFromString(self.inputDescription.label)
        self.layout.addRow(self.label, self.editsWid)
    
    def read(self):
        if self.inputDescription.outputName is None:
            return {}
        l =[] 
        for i in self.inputs:
            l.append(self.enumToType(self.inputDescription.inType)(i.text()))###TODO: incomplete Conversion
        if len(l)>1:        
            if self.inputDescription.oArray:
                l = array(l)
        else:
            l = l[0]
        l = {self.inputDescription.outputName:l}
        return l ###TODO: Fishy stuff here!!!

    @staticmethod
    def enumToType(intype):
        if intype == inType.vectorIntegers: return int
        if intype == inType.vectorFloats: return float
        if intype == inType.vectorStrings: return str



class SimpleRadio(selectionWidgetInterface):

    selectionChangedSignal = pyqtSignal(int)

    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp

        layout = QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)

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


    def read(self):
        if self.inputDescription.outputName is None:
            return {}
        d = {self.inputDescription.outputName : self.inputDescription.options[self.group.checkedId()]}
        print(d)
        return d
    
    def currentIndex(self):
        return self.group.checkedId()

    def selectionChanged(self):
        self.selectionChangedSignal.emit(self.group.checkedId())



class SimpleDropdown(selectionWidgetInterface):
    
    selectionChangedSignal = pyqtSignal(int)

    def __init__ (self, inp:InputDescription, dynamic = False):
        super().__init__()
        self.inputDescription = inp

        layout = QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)


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


    def read(self):
        if self.inputDescription.outputName is None:
            return {}
        d = {self.inputDescription.outputName : self.inputDescription.options[self.comboBox.currentIndex()]}
        print(d)
        return d
    

    def selectionChanged(self):
        self.selectionChangedSignal.emit(self.comboBox.currentIndex())

    def currentIndex(self):
        return self.comboBox.currentIndex()


class XYZRadio(inputWidgetInterface):
    class RadioSubWidget(QWidget):
        def __init__(self, options, name, parent=None):
            super().__init__(parent)
            layout = QGridLayout(self)
            layout.setContentsMargins(0,0,0,0)
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
        
    def __init__(self, inp, parent=None):
        super().__init__(parent)
        self.inputDescription = inp

        layout = QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.r1 = self.RadioSubWidget(["x", "y", "z"],"r1")
        self.r2 = self.RadioSubWidget(["x", "y", "z"],"r2")
        self.r1.setCompanion(self.r2)
        self.r2.setCompanion(self.r1)
        layout.addRow(MyLabel("Abscissa"), self.r1)
        layout.addRow(MyLabel("Ordinate"), self.r2)

    def read(self):
        if self.r1.group.checkedButton()==None or self.r2.group.checkedButton()==None:
            raise Exception("RadioButton no option selected") 
        return {self.inputDescription.outputName:self.r1.group.checkedButton().text() + self.r2.group.checkedButton().text()}
