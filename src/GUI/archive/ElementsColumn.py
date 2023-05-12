from PySide2 import QtWidgets as qtw
from PySide2 import QtCore
from PySide2.QtGui import QFont, QIcon


from src.GUI.ElementWidget import ElementWidget

class ElementsWindow(qtw.QWidget):
    def __init__ (self, elements, actions):
        super().__init__()

        pushDown = 0
        for elem in elements:
            label = ElementWidget(elem, actions, self)
            label.move(0,pushDown)
            pushDown += 40
           
        self.setMaximumHeight(500)
   
        
if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = ElementsWindow(["lala", "lolo"])
    mw.resize(400,300)
    mw.show()
    app.exec_()