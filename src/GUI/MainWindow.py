import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction
from PyQt5.QtGui import QFont, QIcon
from src.GUI.ElementsColumn import ElementsWindow
from src.GUI.SystemsColumn import SystemsWindow
from src.GUI.ParameterForms import ParabolaForm, HyperbolaForm
from src.GUI.PlotScreen import PlotScreen
import numpy as np

sys.path.append('../')
sys.path.append('../../')
import src.POPPy.System as st

class MainWidget(QWidget):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("POPPy")

        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        self.GPSystemsColumn  = [2, 0, 2, 1]
        self.GPButtons        = [2, 0, 1, 1]
        self.GPParameterForm  = [0, 1, 4, 1]
        self.GPPlotScreen     = [0, 2, 4, 1]


        ### ElementConfigurations
        self.elementConfigs = []

        # init System
        self.SystemsList = {}
        



        # init layout
        self.grid = QGridLayout()

        self._mkElementsColumn()
        self._mkSystemsColumn()
        # self._mkButtons()
        # self.grid.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum),3,3)
        self.plotSystem()


        self.setLayout(self.grid)
    
    def _mkElementsColumn(self):
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)
        StmElements = []
        for c in self.elementConfigs:
            StmElements.append(c["name"])
        self.ElementsColumn = ElementsWindow(StmElements)
        self.ElementsColumn.setMaximumWidth(300)
        self.ElementsColumn.setMinimumWidth(300)
        self.addToWindowGrid(self.ElementsColumn, self.GPElementsColumn)

    
    def _mkSystemsColumn(self):
        if hasattr(self, "SystemsColumn"):
            self.SystemsColumn.setParent(None)
        listofstr = []
        for k,_ in self.SystemsList.items():
            listofstr.append(k)
        self.SystemsColumn = SystemsWindow(listofstr)
        self.SystemsColumn.setMaximumWidth(300)
        self.SystemsColumn.setMinimumWidth(300)
        self.addToWindowGrid(self.SystemsColumn, self.GPSystemsColumn)

    def plotSystem(self):
        if hasattr(self, "PlotScreen"):
            self.PlotScreen.setParent(None)

        if self.SystemsList:
            figure, _ = self.SystemsList["System 1"].plotter.plotSystem(self.SystemsList["System 1"].system , ret = True, show=False, save=False)
        else :
            figure = None
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

    def addExampleParabola(self):
        d = {'name': 'parabol', 'type': 'Parabola', 'gridsize': [101, 101], 'flip': False, 'pmode': 'manual', 'coeffs':    np.array[1., 1., 0.], 'gmode': 'xy', 'lims_x': [-1.0, 1.0], 'lims_y': [-1.0, 1.0]}
        self.addElementAction(d)

    def addExampleHyperbola(self):
        d = {'name': 'Hyper', 'type': 'Hyperbola', 'gridsize': [201, 201], 'flip': False, 'pmode': 'manual', 'coeffs': np.array([3., 3., 2.]), 'gmode': 'xy', 'lims_x': [-1.5, 1.5], 'lims_y': [0.5, 0.5]}
        self.addElementAction(d)

    def addElementAction(self, elementDict):
        self.elementConfigs.append(elementDict)
        
        print(self.elementConfigs[-1])
        self._mkElementsColumn()

    def addSystemAction(self):
        stm = st.System()
        stm.addPlotter()

        for elementDict in self.elementConfigs:
            if elementDict["type"] == "Parabola":
                stm.addParabola(elementDict) 
            elif elementDict["type"] == "Hyperbola":
                stm.addHyperbola(elementDict) 

        self.SystemsList["System 1"] = stm
        
        # self.plotSystem()
        self._mkSystemsColumn()

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
        with open('src/GUI/style.css') as f:
            style = f.read()
        self.setStyleSheet(style)


    def _createMenuBar(self):
        menuBar = self.menuBar()

        ElementsMenu = menuBar.addMenu("Elements")
        SystemsMenu = menuBar.addMenu("Systems")

        ### Generate test parabola
        AddTestParabola = QAction('Add Test Parabola', self)
        AddTestParabola.setShortcut('Ctrl+Shift+P')
        AddTestParabola.setStatusTip('Generates a Parabolic reflector and plots it')
        AddTestParabola.triggered.connect(self.mainWid.addExampleParabola)
        ElementsMenu.addAction(AddTestParabola)

        ### Generate test hyperbola
        AddTestHyperbola = QAction('Add Test Hyperbola', self)
        AddTestHyperbola.setShortcut('Ctrl+Shift+H')
        AddTestHyperbola.setStatusTip('Generates a Parabolic reflector and plots it')
        AddTestHyperbola.triggered.connect(self.mainWid.addExampleHyperbola)
        ElementsMenu.addAction(AddTestHyperbola)

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
        newSystem.triggered.connect(self.mainWid.addSystemAction)
        SystemsMenu.addAction(newSystem)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = PoppyMainWindow(parent=None)
    win.show()
    app.exec_()
