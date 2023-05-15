from PySide2.QtWidgets import QLabel, QLineEdit, QPushButton
from PySide2.QtCore import Qt, QRegExp
from PySide2.QtGui import QRegExpValidator
from enum import Enum, auto

##
# Raised when an inputDescription is created incorrectly  
class InputDescriptionError(Exception):
    pass

##
# Enum containing the possible types for an inputDescription 
class inType(Enum):
    static = 0
    vectorStrings = 1
    vectorIntegers = 2
    vectorFloats = 3
    checkbox = 4
    dropdown = 5
    radio = 6
    xyzRadio = 7
    dynamicDropdown = 8
    dynamicRadio = 9
    elementSelector = 10

##
# Enum containing workspace sections  
class WSSections(Enum):
    Element = 0
    RayTraceFrame = 1
    POField = 2
    SPOField = 3
    POCurrent = 4
    Group = 5
    
class MyButton(QPushButton):
    def __init__(self, s):
        super().__init__(s)

class MyLabel(QLabel):
    def __init__ (self, s):
        super().__init__(s)
        self.setWordWrap(True)

class MyEdit(QLineEdit):
    def __init__ (self):
        super().__init__()
        self.setAlignment = Qt.AlignTop

def makeLabelFromString(s):
    return MyLabel(s.replace("_"," ").capitalize())

def getValidator(intype):
    if intype == inType.vectorIntegers:
        return QRegExpValidator(QRegExp("[-+]?[0-9]*"))
    elif intype == inType.vectorFloats:
        return  QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*(e-?)?[0-9]*"))
    elif intype == inType.vectorStrings:
        return  QRegExpValidator(QRegExp("[A-Za-z0-9_]+"))
    else:
        raise Exception("No validator available")