from PySide2.QtWidgets import QWidget
from abc import ABC, abstractclassmethod

##
# @file 
# Defines interfaces for input widgets to inherit from to ensure certain behavior.

##
# Metaclass for inputWidgetInterface because Python does not allow multiple inheritance.
class MetaCombinerInputWidget(type(QWidget), type(ABC)): pass

##
# This is a base class for inputWidgets and should not be instantiated.
class inputWidgetInterface(QWidget, ABC, metaclass=MetaCombinerInputWidget):
    def __init__ (self, parent = None):
        super().__init__(parent)

    ##
    # subclasses need to reimplement this method and return a dict 
    # containing strings as keys. These strings should originate from 
    # the instances InputDescription.outputName. If outputName is 
    # None then it should return an empty dict.
    @abstractclassmethod
    def read() -> dict:
        pass
    
    ##
    # Clears user input and resets widget.
    @abstractclassmethod
    def clear() -> dict:
        pass

##
# Metaclass for selectionWidgetInterface because Python does not allow multiple inheritance.
class MetaCombinerSelectionWidget(type(inputWidgetInterface), type(ABC)): pass

##
# Provides ground work for dynamic input widgets by forcing a selection widget te implement a 
# selectionChanged method. This method might emit a signal to notify its parent that the user 
# has changed the selection.
class selectionWidgetInterface(inputWidgetInterface, ABC, metaclass=MetaCombinerSelectionWidget):
    def __init__ (self, parent = None):
        super().__init__(parent)

    ##
    # Should be reimplemented to emit a signal.
    #
    # @see selectionWidgetInterface
    @abstractclassmethod
    def selectionChanged(self):
        pass
    
