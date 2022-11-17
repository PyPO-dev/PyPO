from PyQt5.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QPushButton, QApplication
from ElementsColumn import ElementsWindow
from PyQt5 import QtCore
from src.GUI.ParameterForms.ParabolaForm import ParabolaFormLayout
import PlotScreen as PS
from src.GUI.TransformationWidget import TransformationWidget
import sys
import numpy as np
from matplotlib.figure import Figure


sys.path.append('../')
sys.path.append('../../')
import POPPy.System as st



class SystemTab(QWidget):
    def __init__ (self):
        super().__init__()

        
        # init System
        self.System = st.System()
        self.Layout = QGridLayout()

        self.buttons = QWidget()
        self.buttonsLayout = QVBoxLayout()
        
        self.addElementBtn = QPushButton("AddElement(parabola)")
        self.addElementBtn.clicked.connect(self.addElement)
        self.buttonsLayout.addWidget(self.addElementBtn)

        self.plotBtn = QPushButton("Plot")
        self.plotBtn.clicked.connect(self.plotSystem)
        self.buttonsLayout.addWidget(self.plotBtn)

        # self.clearBtn = QPushButton("Clear Form")
        # self.clearBtn.clicked.connect(self.clearForm)
        # self.buttonsLayout.addWidget(self.clearBtn)
        
        self.buttons.setLayout(self.buttonsLayout)
        self.buttons.setMaximumWidth(180)
        self.buttons.setMinimumWidth(180)
        self.buttons.setMaximumHeight(80)
        self.buttons.setMinimumHeight(80)
        self.Layout.addWidget(self.buttons, 0,0)


        # ElementsWindow
        self.ElementsWindow = QWidget()
        self.refreshElements()

        # Parameterform
        self.buildParameterForm()
        

        # PlotScreen
        self.PlotScreen = PS.PlotScreen(Figure())
        self.Layout.addWidget(self.PlotScreen, 0, 2, 3, 1)

        self.setLayout(self.Layout)        

    def addElement(self):
        self.parameterformLayout = ParabolaFormLayout(self)
        self.Parameterform.setLayout(self.parameterformLayout)
    # def clearForm(self):
    #     for i in reversed(range(self.parameterformLayout.count())): 
    #         print(i)
    #         print(type(self.parameterformLayout.itemAt(i)))
        
    def plotSystem(self):
        self.PlotScreen.setParent(QWidget())

        for attr in self.System.system['Parabola_0']:
            print(attr)
        figure = self.System.plotSystem(ret = True)
        self.PlotScreen= PS.PlotScreen(figure)
        self.Layout.addWidget(self.PlotScreen, 0, 2, 3, 1)
    
    def refreshElements(self):
        self.ElementsWindow.setParent(QWidget())

        elementArray = ["placeholder"]
        for key, element in self.System.system.items():
            elementArray.append(key)
            print("item added to array")
        self.ElementsWindow = ElementsWindow(elementArray)
        self.ElementsWindow.setMaximumWidth(180)
        self.ElementsWindow.setMinimumWidth(180)
        self.Layout.addWidget(self.ElementsWindow, 1, 0)

    def buildParameterForm(self):
        self.Parameterform = QWidget()
        self.Parameterform.setMaximumWidth(400)
        self.Parameterform.setMinimumWidth(400)

        self.Layout.addWidget(self.Parameterform, 0, 1, 2, 1)

        # TransformationWidget
        self.transformation = TransformationWidget()
        self.transformation.setMaximumWidth(400)
        self.transformation.setMinimumWidth(400)
        self.Layout.addWidget(self.transformation, 2, 1,1,1)

if __name__ == '__main__':

    app = QApplication([])
    mw = SystemTab()
    mw.show()
    app.exec_()