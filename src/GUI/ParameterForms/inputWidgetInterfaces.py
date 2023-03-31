from PyQt5.QtWidgets import QWidget
# from PyQt5.QtCore import pyqtSignal
from abc import ABC, abstractclassmethod

from src.GUI.ParameterForms.InputDescription import InputDescription, inType


##
# @file 
# Defines interfaces for input widgets to inherit from to ensure certain behavior.

class MetaCombinerInputWidget(type(QWidget), type(ABC)): pass
##
# This is a base class for inputWidgets and should not be instantiated
class inputWidgetInterface(QWidget, ABC, metaclass=MetaCombinerInputWidget):
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    ##
    # subclasses need to reimplement this method and return a dict 
    # containing strings as keys. These strings should originate from 
    # the instances InputDescription.outputName. If outputName is 
    # None then it should return an empty dict     
    def read() -> dict:
        pass
    
    @abstractclassmethod
    ##
    # Clears user input and resets widget
    def clear() -> dict:
        pass

class MetaCombinerSelectionWidget(type(inputWidgetInterface), type(ABC)): pass
##
# Provides ground work for dynamic input widgets by forcing a selection widget te implement a 
# selectionChanged method. This method might emit a signal to notify its parent that the user 
# has changed the selection
class selectionWidgetInterface(inputWidgetInterface, ABC, metaclass=MetaCombinerSelectionWidget):
    """This is a base class for selectionWidgets and should not be instantiated"""
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    ##
    # Should be reimplemented to emit a signal
    # @see selectionWidgetInterface
    def selectionChanged(self):
        pass
   


    