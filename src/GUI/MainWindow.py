import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction
from PyQt5.QtGui import QFont, QIcon
from ElementsColumn import ElementsWindow
from GUI.ParameterForms import ParabolaForm, HyperbolaForm
from POPPy.Reflectors import Parabola
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

        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        # self.GPSystemsColumn  = [1, 0, 1, 1]
        self.GPButtons        = [2, 0, 1, 1]
        self.GPParameterForm  = [0, 1, 3, 1]
        self.GPPlotScreen     = [0, 2, 3, 1]


        ### ElementConfigurations
        self.elementConfigs = []

        # init System
        self.STM = st.System()
        self.STM.addPlotter()
        # self.STM.addParabola([1,1], [-1,1], [-1,1], [101,101]) ##TODO: remove 



        # init layout
        self.grid = QGridLayout()

        self._mkElementsColumn()
        # self._mkButtons()
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

    def plotSystem(self):
        if hasattr(self, "PlotScreen"):
            self.PlotScreen.setParent(None)

        figure = self.STM.plotter.plotSystem(ret = True, show=False)
        self.PlotScreen= PlotScreen(figure)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])


    # def _mkButtons(self):
    #     btn = QPushButton("addParabola")
    #     btn.clicked.connect(self.btnAction)

    #     btnWidget = QWidget()
    #     btnLayout = QVBoxLayout()
    #     btnLayout.addWidget(btn)
    #     btnWidget.setLayout(btnLayout)
    #     self.addToWindowGrid(btnWidget,self.GPButtons)

    def btnAction(self):
        self.STM.addParabola([1,1], [-1,1], [-1,1], [101,101]) 
        print("Parabola added!!!")
        self.plotSystem()
        self._mkElementsColumn()

    def addElementAction(self, elementDict):
        self.elementConfigs.append(elementDict)
        self.STM.addParabola(elementDict) 
        print(self.elementConfigs[-1])

    def setParabolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = ParabolaForm.Form(self.addElementAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def setHyperbolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = HyperbolaForm.Form(self.addElementAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)


class PoppyMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.mainWid = MainWidget()
        self.setAutoFillBackground(True)
        self._createMenuBar()
        self.setCentralWidget(self.mainWid)
        self.showMaximized()
        with open('style.css') as f:
            style = f.read()
        self.setStyleSheet(style)


    def _createMenuBar(self):
        menuBar = self.menuBar()

        ElementsMenu = menuBar.addMenu("Elements")
        SystemsMenu = menuBar.addMenu("Systems")

        ### Generate test parabola
        AddTestElement = QAction('Add Test Parabola', self)
        AddTestElement.setShortcut('Ctrl+A')
        AddTestElement.setStatusTip('Generates a Parabolic reflector and plots it')
        AddTestElement.triggered.connect(self.mainWid.btnAction)
        ElementsMenu.addAction(AddTestElement)

        ### Add Element
        newElementMenu = ElementsMenu.addMenu("New Element")
        reflectorSelector = newElementMenu.addMenu("Reflector")
        ### Parabola
        parabolaAction = QAction('Parabola', self)
        parabolaAction.setShortcut('Ctrl+P')
        parabolaAction.setStatusTip("Add a parabolic Reflector")
        parabolaAction.triggered.connect(self.mainWid.setParabolaForm)
        reflectorSelector.addAction(parabolaAction)
        ### Hyperbola
        hyperbolaAction = QAction('Hyperbola', self)
        hyperbolaAction.setShortcut('Ctrl+H')
        hyperbolaAction.setStatusTip("Add a parabolic Reflector")
        hyperbolaAction.triggered.connect(self.mainWid.setHyperbolaForm)
        reflectorSelector.addAction(hyperbolaAction)
        ### Ellipse
        

    ### System actions
        newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid...)
        SystemsMenu.addAction(newSystem)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = PoppyMainWindow(parent=None)
    win.show()
    app.exec_()
