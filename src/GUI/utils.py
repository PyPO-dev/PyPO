"""!
@file
Contains utilities for the GUI. 
"""

from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton
from PySide6.QtCore import Qt, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from enum import Enum, auto

class InputDescriptionError(Exception):
    """!
    Raised when an inputDescription is created incorrectly  
    """
    pass

class inType(Enum):
    """!
    Enum containing the possible types for an inputDescription 
    """
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

class WSSections(Enum):
    """!
    Enum containing workspace sections  
    """
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
        return QRegularExpressionValidator(QRegularExpression("[-+]?[0-9]*"))
    elif intype == inType.vectorFloats:
        return  QRegularExpressionValidator(QRegularExpression("[-+]?[0-9]*[\.,]?[0-9]*(e-?)?[0-9]*"))
    elif intype == inType.vectorStrings:
        return  QRegularExpressionValidator(QRegularExpression("[A-Za-z0-9_]+"))
    else:
        raise Exception("No validator available")
