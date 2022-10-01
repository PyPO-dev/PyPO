import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QFont, QIcon
from ElementsColumn import ElementsWindow
from PlotScreen import PlotScreen
from ParameterForms.BaseForm import FormWidget

sys.path.append('../')
sys.path.append('../../')
import POPPy.System as st

class MainWidget(QWidget):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("POPPy")
        # self._createMenuBar()

        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        self.GPButtons =        [2, 0, 1, 1]
        self.GPParameterForm =  [0, 1, 3, 1]
        self.GPPlotScreen =     [0, 2, 3, 1]


        ### ElementConfigurations
        

        # init System
        self.STM = st.System()
        # self.STM.addParabola([1,1], [-1,1], [-1,1], [101,101]) ##TODO: remove 



        # init layout
        self.grid = QGridLayout()

        self._mkElementsColumn()
        self._mkParameterWidget()
        self._mkButtons()
        # self.grid.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum),3,3)
        self.plotSystem()


        self.setLayout(self.grid)
        
        
    def _mkElementsColumn(self):
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)
        StmElements = []
        for key, element in self.STM.system.items():
            StmElements.append(key)
        self.ElementsColumn = ElementsWindow(StmElements)
        self.ElementsColumn.setMaximumWidth(300)
        self.ElementsColumn.setMinimumWidth(300)
        self.addToWindowGrid(self.ElementsColumn, self.GPElementsColumn)
        
    def _mkParameterWidget(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = FormWidget()
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        # self.ParameterWid.setProperty('class', 'parameterForm')
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def plotSystem(self):
        if hasattr(self, "PlotScreen"):
            self.PlotScreen.setParent(None)

        figure = self.STM.plotSystem(ret = True)
        self.PlotScreen= PlotScreen(figure)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])


    def _createMenuBar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)

        ElementsMenu = QMenu("Elements", self)
        menuBar.addMenu(ElementsMenu)

    def _mkButtons(self):
        btn = QPushButton("addParabola")
        btn.clicked.connect(self.btnAction)

        btnWidget = QWidget()
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(btn)
        btnWidget.setLayout(btnLayout)
        self.addToWindowGrid(btnWidget,self.GPButtons)

    def btnAction(self):
        self.STM.addParabola([1,1], [-1,1], [-1,1], [101,101]) 
        print("Parabola added!!!")
        self.plotSystem()
        self._mkElementsColumn()

    # def removeWidget(self, widget):
    #     if widget!= None:
    #         widget.setParent(None)




    def AddElementAction(self):
        pass








if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QMainWindow(parent=None)
    win.showMaximized()
    win.setCentralWidget(MainWidget())

    with open('style.css') as f:
        css = f.read()
    win.setStyleSheet(css)

    win.show()
    app.exec_()
