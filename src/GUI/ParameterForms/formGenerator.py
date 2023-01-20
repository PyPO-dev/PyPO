from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QLabel, QLineEdit, QHBoxLayout, QPushButton, QStackedWidget, QCheckBox, QRadioButton, QButtonGroup, QGridLayout
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

class SimpleRadio(QWidget):
    def __init__ (self, inp:InputDescription):
        super().__init__()
        self.inputDescription = inp

        layout = QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)

        radioWidget = QWidget()
        radiolayout = QHBoxLayout(radioWidget)

        self.group = QButtonGroup()
        for i in self.inputDescription.sublist:
            rb = QRadioButton(i)
            self.group.addButton(rb)
            radiolayout.addWidget(rb)
        layout.addRow(MyLabel(self.inputDescription.label) ,radioWidget)

    def read(self):
        d = {self.inputDescription.outputName:self.group.checkedButton().text()}
        print(d)
        return d





class XYZRadio(QWidget):


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
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QFormLayout(self)
        layout.setContentsMargins(0,0,0,0)
        self.r1 = self.RadioSubWidget(["x", "y", "z"],"r1")
        self.r2 = self.RadioSubWidget(["x", "y", "z"],"r2")
        self.r1.setCompanion(self.r2)
        self.r2.setCompanion(self.r1)
        layout.addRow(QLabel("Horizontal axis"), self.r1)
        layout.addRow(QLabel("Vertical axis"), self.r2)
        btn = QPushButton("Read")
        btn.clicked.connect(self.read)
        layout.addRow(btn)

    def read(self):
        if self.r1.group.checkedButton()==None or self.r2.group.checkedButton()==None:
            raise Exception("RadioButton no option selected") 
        print(self.r1.group.checkedButton().text()+self.r2.group.checkedButton().text())

        def uncheckOthers(self, caller):
            for btn in self.buttons:
                if btn is not caller:
                    btn.setChecked(False)

        def toggled(self, b):
            btn = self.group.checkedButton()
            print(type(btn))
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
                    print("@uncheckOption %s btn = %s, s = %s, " %(self.name,btn.text(), s))
                    return
    