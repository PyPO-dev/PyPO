"""!
@file 
Defines interfaces for input widgets to inherit from to ensure certain behavior.
"""

from PySide6.QtWidgets import QWidget
from abc import ABC, abstractclassmethod

class MetaCombinerInputWidget(type(QWidget), type(ABC)): 
    """!
    Metaclass for inputWidgetInterface because Python does not allow multiple inheritance.
    """
    pass

class inputWidgetInterface(QWidget, ABC, metaclass=MetaCombinerInputWidget):
    """!
    This is a base class for inputWidgets and should not be instantiated.
    """
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    def read() -> dict:
        """!
        subclasses need to reimplement this method and return a dict 
        containing strings as keys. These strings should originate from 
        the instances InputDescription.outputName. If outputName is 
        None then it should return an empty dict.
        """
        pass
    
    @abstractclassmethod
    def clear() -> dict:
        """!
        Clears user input and resets widget.
        """
        pass


class MetaCombinerSelectionWidget(type(inputWidgetInterface), type(ABC)): 
    """!
    Metaclass for selectionWidgetInterface because Python does not allow multiple inheritance.
    """
    pass


class selectionWidgetInterface(inputWidgetInterface, ABC, metaclass=MetaCombinerSelectionWidget):
    """!
    Provides ground work for dynamic input widgets by forcing a selection widget te implement a 
    selectionChanged method. This method might emit a signal to notify its parent that the user 
    has changed the selection.
    """
    def __init__ (self, parent = None):
        super().__init__(parent)

    @abstractclassmethod
    def selectionChanged(self):
        """!
        Should be reimplemented to emit a signal.
        
        @see selectionWidgetInterface
        """
        pass
    
