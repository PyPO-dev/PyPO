from PyQt5.QtWidgets import QWidget
# from PyQt5.QtCore import pyqtSignal
from abc import ABC, abstractclassmethod

from src.GUI.ParameterForms.InputDescription import InputDescription, inType

class MetaCombinerInputWidget(type(QWidget), type(ABC)): pass
class inputWidgetInterface(QWidget, ABC, metaclass=MetaCombinerInputWidget):
    """This is a base class for inputWidgets and should not be instatiated"""
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    def read() -> dict:
        """
        subclasses need to reimplement this method and return a dict 
        containing a string as key. This string should originate from 
        the instances InputDescription.outputName. If outputName is 
        None then it should return an empty dict   
        """
        pass

    @abstractclassmethod
    def clear() -> dict:
        pass

class MetaCombinerselectionWidget(type(inputWidgetInterface), type(ABC)): pass
class selectionWidgetInterface(inputWidgetInterface, ABC, metaclass=MetaCombinerselectionWidget):
    """This is a base class for selectionWidgets and should not be instatiated"""
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    def selectionChanged(self):
        raise NotImplementedError
   


    