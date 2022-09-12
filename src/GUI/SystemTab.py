from PyQt5 import QtWidgets as qtw
# from .. Python.System import System
from ElementsColumn import ElementsWindow
from PyQt5 import QtCore
from ParabolaForm import ParabolaFormLayout

class SystemTab(qtw.QWidget):
    def __init__ (self):
        super().__init__()
        self.Elements = ["parabola", "Ellipse"]
        
        # # TODO: init System
        # self.System = System()
        
        self.Layout = qtw.QHBoxLayout()

        self.ElementSelector = qtw.QComboBox()
        self.ElementSelector.setStyleSheet("background: green; font-size: 16;")
        self.ElementSelector.addItems(["Select Element..","Reflector","Camera","Input Beam","Ray Tracer","TODO"])
        self.ElementSelector.activated.connect(self.selectionChanged)
        # addElementBtn.clicked.connect(lambda:self.addElement())

        # ElementsWindow
        self.ElementsWindow = ElementsWindow(self.Elements, self.ElementSelector)
        self.ElementsWindow.setMaximumWidth(180)
        self.ElementsWindow.setMinimumWidth(180)
        self.Layout.addWidget(self.ElementsWindow,0)

        # Parameterform
        self.Parameterform = qtw.QWidget()
        self.Parameterform.setMaximumWidth(400)
        self.Parameterform.setMinimumWidth(400)

        self.Parameterform.setStyleSheet("background: #D9D9D9")
        # 
        self.Layout.addWidget(self.Parameterform,0)


        # self.setStyleSheet("background: #5A0168; color:white")

        self.setLayout(self.Layout)        

    def selectionChanged(self):
        if self.ElementSelector.currentIndex() == 1:
            self.Parameterform.setLayout(ParabolaFormLayout())
        self.ElementSelector.setCurrentIndex(0)
        


if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = SystemTab()
    mw.show()
    app.exec_()