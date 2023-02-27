from PyQt5.QtWidgets import QWidget
from abc import ABC, abstractclassmethod

from src.GUI.ParameterForms.InputDescription import InputDescription, inType

class MetaCombiner(type(QWidget), type(ABC)): pass

class inputWidgetInterface(QWidget, ABC, metaclass=MetaCombiner):
    """This is a base class for inputWidgets and should not be instatiated"""
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    def read() -> dict:
        """
        subclasses need to reimplement this method and return a dict 
        containing a string as key. This string should orinate from 
        the instances InputDescription.outputName. If outputName is 
        None then it should return an empty dict   
        """
        pass