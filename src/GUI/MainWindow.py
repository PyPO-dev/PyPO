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

    def addElementAction(self):
        elementDict = self.ParameterWid.read()
        if elemType == "Parabola":
            self.stm.addParabola(elementDict) 
        elif elemType == "Hyperbola":
            self.stm.addHyperbola(elementDict) 
        elif elemType == "Plane":
            self.stm.addPlane(elementDict) 
        
        self._mkElementsColumn()

 

    def setHyperbolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.makeHyperbolaEllipseInp(), self.addHyperbolaAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def addHyperbolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addHyperbola(elementDict) 
        self._mkElementsColumn()
    
    def setEllipseForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.makeHyperbolaEllipseInp(), self.addEllipseAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def addEllipseAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addEllipse(elementDict) 
        self._mkElementsColumn()
    
    def setParabolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.makeParabolaInp(), self.addParabolaAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def addParabolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addParabola(elementDict) 
        self._mkElementsColumn()
    """
    def setHyperbolaForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = HyperbolaForm.Form(self.addElementAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)
    """
    def setPlaneForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.makePlaneInp(), self.addPlaneAction)        
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)
    
    def addPlaneAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addPlane(elementDict) 
        self._mkElementsColumn()
    
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

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.initFrameInp(), self.addFrameAction)
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

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.plotFrameInp(self.frameDict), self.addPlotFrameAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid, self.GPParameterForm)

    def addPlotFrameAction(self):
        if hasattr(self, "PlotScreen"):
            self.PlotScreen.setParent(None)
        
        plotFrameDict = self.ParameterWid.read()
        fig = self.stm.plotRTframe(self.frameDict[plotFrameDict["frame"]], project=plotFrameDict["project"], returns=True)
        self.PlotScreen = PlotScreen(fig)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def setPropRaysForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)

        self.ParameterWid = formGenerator.FormGenerator(fDataObj.propRaysInp(self.frameDict, self.stm.system), self.addPropRaysAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid, self.GPParameterForm)

    def addPropRaysAction(self): 
        propRaysDict = self.ParameterWid.read()
        frame_out = self.stm.runRayTracer(self.frameDict[propRaysDict["frame_in"]], 
                                    propRaysDict["target"], propRaysDict["epsilon"], propRaysDict["nThreads"], 
                                    propRaysDict["t0"], propRaysDict["device"], verbose=False)
        name = f"frame_{len(self.frameDict)}"
        self.frameDict[name] = frame_out
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
        PhysOptMenu = menuBar.addMenu("Physical optics")

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
        parabolaAction = QAction('Paraboloid', self)
        parabolaAction.setShortcut('Ctrl+P')
        parabolaAction.setStatusTip("Add a paraboloid Reflector")
        parabolaAction.triggered.connect(self.mainWid.setParabolaForm)
        reflectorSelector.addAction(parabolaAction)
        ### Hyperbola
        hyperbolaAction = QAction('Hyperboloid', self)
        hyperbolaAction.setShortcut('Ctrl+H')
        hyperbolaAction.setStatusTip("Add a hyperboloid Reflector")
        hyperbolaAction.triggered.connect(self.mainWid.setHyperbolaForm)
        reflectorSelector.addAction(hyperbolaAction)
        ### Ellipse
        ellipseAction = QAction('Ellipsoid', self)
        ellipseAction.setShortcut('Ctrl+H')
        ellipseAction.setStatusTip("Add an ellipsoid Reflector")
        ellipseAction.triggered.connect(self.mainWid.setEllipseForm)
        reflectorSelector.addAction(ellipseAction)
        ### Plane 
        planeAction = QAction("Plane", self)
        planeAction.setShortcut("Ctrl+L")
        planeAction.setStatusTip("Add a plane surface.")
        planeAction.triggered.connect(self.mainWid.setPlaneForm)
        reflectorSelector.addAction(planeAction)

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

        # Propagate rays
        # propRaysAction = QAction("Propagate rays", self)
        # propRaysAction.triggered.connect(self.mainWid.setPropRaysForm)
        # RaytraceMenu.addAction(propRaysAction)

        # PO actions
        makeBeam = PhysOptMenu.addMenu("Initialize beam")
        initPointAction = QAction("Point source", self)
        makeBeam.addAction(initPointAction)

        # END NOTE
if __name__ == "__main__":

    app = QApplication(sys.argv)
    win = PyPOMainWindow(parent=None)
    win.show()
    app.exec_()
