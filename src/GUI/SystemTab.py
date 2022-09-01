from PyQt5 import QtWidgets as qtw
# from .. Python.System import System
from ElementsColumn import ElementsWindow
from PyQt5 import QtCore
# from ParabolaForm import ParabolaForm

class SystemTab(qtw.QWidget):
    def __init__ (self):
        super().__init__()
        self.Elements = ["parabola", "Ellipse"]
        
        # # TODO: init System
        # self.System = System()
        
        self.Layout = qtw.QHBoxLayout()

        ElementSelector = qtw.QComboBox()
        ElementSelector.setStyleSheet("background: green; font-size: 16;")
        ElementSelector.addItems(["Reflector","Camera","Input Beam","Ray Tracer","TODO"])
        # addElementBtn.clicked.connect(lambda:self.addElement())

        # ElementsWindow
        self.ElementsWindow = ElementsWindow(self.Elements, ElementSelector)
        self.ElementsWindow.setMaximumWidth(180)
        self.Layout.addWidget(self.ElementsWindow)

        # Parameterform
        # self.Parameterform = ParabolaForm()
        # self.Parameterform.setStyleSheet("background: #D9D9D9")
        # self.Parameterform.setMaximumWidth(300)
        # self.Layout.addWidget(self.Parameterform)

        self.setLayout(self.Layout)        

    def addElement(self):
        self.Elements.append("element")
        print("element added!!!")


if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = SystemTab()
    mw.show()
    app.exec_()