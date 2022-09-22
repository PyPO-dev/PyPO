from PyQt5 import QtWidgets as qtw
# from .. Python.System import System
from ElementsColumn import ElementsWindow
from PyQt5 import QtCore
from ParabolaForm import ParabolaFormLayout
import PlotScreen as PS
import sys
import numpy as np
from matplotlib.figure import Figure


sys.path.append('../')
sys.path.append('../../')
import POPPy.System as system



class SystemTab(qtw.QWidget):
    def __init__ (self):
        super().__init__()
        
        # init System
        self.System = system.System()
        # self.PlotScreen 
        self.Layout = qtw.QHBoxLayout()

        self.buttons = qtw.QWidget()
        self.buttonsLayout = qtw.QVBoxLayout()
        

        
        self.addElementBtn = qtw.QPushButton("AddElement(parabola)")
        self.addElementBtn.clicked.connect(self.addElement)
        self.buttonsLayout.addWidget(self.addElementBtn)

        self.plotBtn = qtw.QPushButton("Plot")
        self.plotBtn.clicked.connect(self.plotSystem)
        self.buttonsLayout.addWidget(self.plotBtn)

        # self.clearBtn = qtw.QPushButton("Clear Form")
        # self.clearBtn.clicked.connect(self.clearForm)
        # self.buttonsLayout.addWidget(self.clearBtn)
        
        self.buttons.setLayout(self.buttonsLayout)


        # ElementsWindow
        self.ElementsWindow = ElementsWindow([], self.buttons)
        self.ElementsWindow.setMaximumWidth(180)
        self.ElementsWindow.setMinimumWidth(180)
        self.Layout.addWidget(self.ElementsWindow,0)

        # Parameterform
        self.Parameterform = qtw.QWidget()
        self.Parameterform.setMaximumWidth(400)
        self.Parameterform.setMinimumWidth(400)

        # self.Parameterform.setStyleSheet("background: #D9D9D9")
        self.Layout.addWidget(self.Parameterform,0)

        # PlotScreen
        self.PlotScreen = PS.PlotScreen(Figure())
        self.Layout.addWidget(self.PlotScreen)

        self.setLayout(self.Layout)        

    def addElement(self):
        self.parameterformLayout = ParabolaFormLayout(self.System)
        self.Parameterform.setLayout(self.parameterformLayout)
    # def clearForm(self):
    #     for i in reversed(range(self.parameterformLayout.count())): 
    #         print(i)
    #         print(type(self.parameterformLayout.itemAt(i)))
        
    def plotSystem(self):
        self.PlotScreen.setParent(qtw.QWidget())

        for attr in self.System.system['Parabola_0']:
            print(attr)
        figure = self.System.plotSystem(ret = True)
        self.PlotScreen= PS.PlotScreen(figure)
        self.Layout.addWidget(self.PlotScreen)

if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = SystemTab()
    mw.show()
    app.exec_()