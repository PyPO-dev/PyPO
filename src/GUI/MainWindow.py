import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction
from PyQt5.QtGui import QFont, QIcon
from src.GUI.ElementsColumn import ElementsWindow
from src.GUI.archive.SystemsColumn import SystemsWindow
from src.GUI.ParameterForms import ParabolaForm, HyperbolaForm, formGenerator
import src.GUI.ParameterForms.formData as fData
import src.GUI.ParameterForms.formDataObjects as fDataObj
from src.GUI.PlotScreen import PlotScreen
from src.GUI.TransformationWidget import TransformationWidget
import numpy as np

sys.path.append('../')
sys.path.append('../../')
import src.PyPO.System as st

class MainWidget(QWidget):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("PyPO")

        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        self.GPSystemsColumn  = [2, 0, 2, 1]
        self.GPButtons        = [2, 0, 1, 1]
        self.GPParameterForm  = [0, 1, 4, 1]
        self.GPPlotScreen     = [0, 2, 4, 1]


        ### ElementConfigurations
        # self.elementConfigs = []

        # init System
        self.stm = st.System()
        



        # init layout
        self.grid = QGridLayout()

        self._mkElementsColumn()
        self.plotSystem()


        self.setLayout(self.grid)

        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE
    
    def _mkElementsColumn(self):
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)
        StmElements = []
        for e,_ in self.stm.system.items():
            StmElements.append(e)
        self.ElementsColumn = ElementsWindow(StmElements, [self.setTransromationForm])
        self.ElementsColumn.setMaximumWidth(300)
        self.ElementsColumn.setMinimumWidth(300)
        self.addToWindowGrid(self.ElementsColumn, self.GPElementsColumn)

    
    def plotSystem(self):
        if hasattr(self, "PlotScreen"):
            self.PlotScreen.setParent(None)

        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False)
        else :
            figure = None
        self.PlotScreen= PlotScreen(figure)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])


    def addExampleParabola(self):
        d = {
            "name"      : "pri",
            "type"      : "Parabola",
            "pmode"     : "focus",
            "gmode"     : "uv",
            "flip"      : False,
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0,0,3.5e3]),
            "lims_u"    : np.array([200,5e3]),
            "lims_v"    : np.array([0,360]),
            "gridsize"  : np.array([1501,1501])
            }
        self.addElementAction(d)

    def addExampleHyperbola(self):
        hyperbola = {
            'type': 'Hyperbola', 
            'pmode': 'focus',
            'gmode': 'xy',
            'flip': False,
            'focus_1': np.array([   0.,    0., 3500.]),
            'focus_2': np.array([    0.,     0., -2106.]),
            'ecc': 1.08208248, 
            'lims_x': np.array([-310, 310]),
            'lims_y': np.array([-310, 310]), 
            'lims_u': np.array([0, 310]), 
            'lims_v': np.array([0, 6.283185307179586]),
            'gridsize': np.array([501, 501])
        }
        self.addElementAction(hyperbola)

    def addElementAction(self, elementDict):
        if elementDict["type"] == "Parabola":
            self.stm.addParabola(elementDict) 
        elif elementDict["type"] == "Hyperbola":
            self.stm.addHyperbola(elementDict) 
        
        self._mkElementsColumn()

 

    def setParabolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fData.parabola,self.addElementAction)
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

    def setTransromationForm(self, element):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)
        
        self.ParameterWid = TransformationWidget(element, self.applyTransformation)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def applyTransformation(self, element, transformationType, transformation, rotationCenter=None):
        # for i in transformation:
        #     print(i)
        # print(transformation, type(transformation))
        # print(rotationCenter, type(rotationCenter))
        if transformationType == "trans":
            self.stm.translateGrids(element, transformation)
        elif transformationType == "rot":
            self.stm.rotateGrids(element, transformation, cRot=rotationCenter)

    #NOTE Raytrace widgets
    def setInitFrameForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.RTGen, self.addFrameAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid, self.GPParameterForm)

    def addFrameAction(self):
        RTDict = self.ParameterWid.read()
        _frame = self.stm.createFrame(RTDict)
        name = f"frame_{len(self.frameDict)}"
        self.frameDict[name] = _frame
    
    def setPlotFrameForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.plotFrameInp(self.frameDict), self.addPlotAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid, self.GPParameterForm)

    def addPlotAction(self):
        RTDict = self.ParameterWid.read()
        _frame = self.stm.createFrame(RTDict)
        name = f"frame_{len(self.frameDict)}"
        self.frameDict[name] = _frame

    #END NOTE

class PyPOMainWindow(QMainWindow):
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
        RaytraceMenu = menuBar.addMenu("Ray-tracer")

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
        # newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid.addSystemAction)
        # SystemsMenu.addAction(newSystem)

        plotSystem = QAction("Plot System", self)
        plotSystem.triggered.connect(self.mainWid.plotSystem)
        SystemsMenu.addAction(plotSystem)

        # NOTE Raytrace actions
        makeFrame = RaytraceMenu.addMenu("Make frame")
        initFrameAction = QAction("Initialize", self)
        initFrameAction.setStatusTip("Initialize ray-trace frame from input form")
        initFrameAction.triggered.connect(self.mainWid.setInitFrameForm)
        makeFrame.addAction(initFrameAction)
        
        poyntingFrameAction = QAction("Poynting", self)

        # Plot frames
        plotFrameAction = QAction("Plot frame", self)
        plotFrameAction.triggered.connect(self.mainWid.setPlotFrameForm)
        RaytraceMenu.addAction(plotFrameAction)

        # END NOTE
if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = PyPOMainWindow(parent=None)
    win.show()
    app.exec_()
