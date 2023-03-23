from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from enum import Enum, auto

# class validation(Enum):
#     integers = auto()

class inType(Enum):
    static = 0
    vectorStrings = 1
    vectorIntegers = 2
    vectorFloats = 3
    checkbox = 4
    dropdown = 5
    radio = 6
    xyzradio = 7
    dynamicDropdown = 8
    dynamicRadio = 9
    elementSelector = 10


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