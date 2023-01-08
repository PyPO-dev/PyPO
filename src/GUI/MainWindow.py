import sys

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar
from PyQt5.QtGui import QFont, QIcon
from src.GUI.ParameterForms import formGenerator
import src.GUI.ParameterForms.formDataObjects as fDataObj
from src.GUI.PlotScreen import PlotScreen
from src.GUI.TransformationWidget import TransformationWidget
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ElementWidget
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
        self._setupPlotScreen()


        self.setLayout(self.grid)

        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE
    
    def _mkElementsColumn(self):
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)

        self.refletorActions = [self.setTransromationForm, self.plotElement, self.removeElement]
        StmElements = []
        for e,_ in self.stm.system.items():
            StmElements.append(e)
        self.ElementsColumn = Accordion()
        self.addToWindowGrid(self.ElementsColumn, self.GPElementsColumn)

    # #####################
    def _setupPlotScreen(self):
        self.PlotScreen = QTabWidget()
        self.PlotScreen.setTabsClosable(True)
        self.PlotScreen.setTabShape(QTabWidget.Rounded)
        self.PlotScreen.tabCloseRequested.connect(self.closeTab)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def addPlot(self, figure, label):
        self.PlotScreen.addTab(PlotScreen(figure), label)
        self.PlotScreen.setCurrentIndex(self.PlotScreen.count()-1)


    def closeTab(self, i):
        self.PlotScreen.removeTab(i)

    def removeElement(self, element):
        self.stm.removeElement(element)
        
    def plotElement(self, surface):
        if self.stm.system:
            figure = self.stm.plot3D(  surface,   returns= True, show=False, save=False, ret=True)
        else :
            figure = None
        self.addPlot(figure, surface)

    def plotSystem(self):
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False)
        else :
            figure = None
        self.addPlot(figure, "System Plot %d" %self.getSysPlotNr())
        
    
    def getSysPlotNr(self):
        if not hasattr(self, "sysPlotNr"):
            self.sysPlotNr = 0
        self.sysPlotNr+=1
        return self.sysPlotNr

    def getRayPlotNr(self):
        if not hasattr(self, "rayPlotNr"):
            self.rayPlotNr = 0
        self.rayPlotNr+=1
        return self.rayPlotNr

    def plotRaytrace(self):
        framelist = []
        if self.stm.frames:
            for val in self.stm.frames.values():
                framelist.append(val)

        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False, RTframes=framelist)
        else :
            figure = None
        self.addPlot(figure,"Ray Trace Frame %d" %(self.getRayPlotNr()))

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
        self.stm.addParabola(d)
        self.ElementsColumn.reflectors.addWidget(ElementWidget(d["name"],self.refletorActions))

    def addExampleHyperbola(self):
        hyperbola = {
            'name': 'Hype',
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
        self.stm.addHyperbola(hyperbola)
        self.ElementsColumn.reflectors.addWidget(ElementWidget(hyperbola["name"],self.refletorActions))

    def setForm(self, formData, readAction):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)
        self.ParameterWid = formGenerator.FormGenerator(formData, readAction)
        self.ParameterWid.setMaximumWidth(400)
        self.ParameterWid.setMinimumWidth(400)
        self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def setQuadricForm(self):
        self.ParameterWid = self.setForm(fDataObj.makeQuadricSurfaceInp(), readAction=self.addParabolaAction)
    
    def setPlaneForm(self):
        self.setForm(fDataObj.makePlaneInp(), readAction=self.addPlaneAction)

    def addHyperbolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addHyperbola(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))

    def addEllipseAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addEllipse(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
    

    def addParabolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addParabola(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
    
    def addPlaneAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addPlane(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
    
    def setTransromationForm(self, element):
        self.setForm(fDataObj.makeTransformationForm(element), self.applyTransformation)
        # if hasattr(self, "ParameterWid"):
        #     self.ParameterWid.setParent(None)
        
        # self.ParameterWid = TransformationWidget(element, self.applyTransformation)
        # self.ParameterWid.setMaximumWidth(400)
        # self.ParameterWid.setMinimumWidth(400)
        # self.addToWindowGrid(self.ParameterWid,self.GPParameterForm)

    def applyTransformation(self, element, transformationType, transformation, rotationCenter=None):

        # for i in transformation:
        print("Transform")
        # print(transformation, type(transformation))
        # print(rotationCenter, type(rotationCenter))
        if transformationType == "trans":
            self.stm.translateGrids(element, transformation)
        elif transformationType == "rot":
            self.stm.rotateGrids(element, transformation, cRot=rotationCenter)

    #NOTE Raytrace widgets
    def setInitFrameForm(self):
        self.setForm(fDataObj.initFrameInp(), readAction=self.addFrameAction)

    def addFrameAction(self):
        RTDict = self.ParameterWid.read()
        self.stm.createFrame(RTDict)
    
    def setPlotFrameForm(self):
        self.setForm(fDataObj.plotFrameInp(self.stm.frames), readAction=self.addPlotFrameAction)

    def addPlotFrameAction(self):
        print(self.stm.frames)
        plotFrameDict = self.ParameterWid.read()
        fig = self.stm.plotRTframe(plotFrameDict["frame"], project=plotFrameDict["project"], returns=True)
        self.PlotScreen.addTab(PlotScreen(fig),"Plot1")

        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def setPropRaysForm(self):
        self.setForm(fDataObj.propRaysInp(self.stm.frames, self.stm.system), self.addPropRaysAction)

    def addPropRaysAction(self): 
        propRaysDict = self.ParameterWid.read()
        self.stm.runRayTracer(propRaysDict["frame_in"], propRaysDict["frame_out"], 
                            propRaysDict["target"], propRaysDict["epsilon"], propRaysDict["nThreads"], 
                            propRaysDict["t0"], propRaysDict["device"], verbose=False)
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
        reflectorSelector = ElementsMenu.addMenu("Reflector")
        ### Planar Surface
        planeAction = QAction("Plane", self)
        planeAction.setShortcut("Ctrl+L")
        planeAction.setStatusTip("Add a plane surface.")
        planeAction.triggered.connect(self.mainWid.setPlaneForm)
        reflectorSelector.addAction(planeAction)
        
        ### Quadric Surface
        hyperbolaAction = QAction('Quadric Surface', self)
        hyperbolaAction.setShortcut('Ctrl+Q')
        hyperbolaAction.setStatusTip("Quadric Surface")
        hyperbolaAction.triggered.connect(self.mainWid.setQuadricForm)
        reflectorSelector.addAction(hyperbolaAction)
        

    ### System actions
        # newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid.addSystemAction)
        # SystemsMenu.addAction(newSystem)

        plotSystem = QAction("Plot System", self)
        plotSystem.triggered.connect(self.mainWid.plotSystem)
        SystemsMenu.addAction(plotSystem)

        plotRaytrace = QAction("Plot ray-trace", self)
        plotRaytrace.triggered.connect(self.mainWid.plotRaytrace)
        SystemsMenu.addAction(plotRaytrace)

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
        propRaysAction = QAction("Propagate rays", self)
        propRaysAction.triggered.connect(self.mainWid.setPropRaysForm)
        RaytraceMenu.addAction(propRaysAction)

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
