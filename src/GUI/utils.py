from PyQt5.QtWidgets import QLabel, QLineEdit
from PyQt5.QtCore import Qt
from enum import Enum



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