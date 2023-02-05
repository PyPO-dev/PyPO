import os
import sys
import shutil
import asyncio

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms import formGenerator
import src.GUI.ParameterForms.formDataObjects as fDataObj
from src.GUI.PlotScreen import PlotScreen
from src.GUI.TransformationWidget import TransformationWidget
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ElementWidget, FrameWidget, FieldsWidget, CurrentWidget
from src.GUI.Console import ConsoleGenerator
from src.GUI.Console import print
import numpy as np
from src.PyPO.Checks import InputReflError, InputRTError

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
        self.GPParameterForm  = [0, 1, 2, 1]
        self.GPPlotScreen     = [0, 2, 1, 1]
        self.GPConsole        = [1, 2, 1, 1]

        ### ElementConfigurations
        # self.elementConfigs = []

        # init System

        
        # init layout
        self.grid = QGridLayout()

        self.pyprint = print

        self._setupPlotScreen()
        self._mkConsole()

        self.stm = st.System(redirect=print)
        self._mkElementsColumn()

        self.setLayout(self.grid)

        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE

    def _mkConsole(self):
        # cons = Console()
        # global print
        # def print(s):
        #     cons.appendPlainText(s)
        # self.console = cons
        self.addToWindowGrid(ConsoleGenerator.get(), self.GPConsole)
        self.console = ConsoleGenerator.get()
        self.cursor = QTextCursor(self.console.document())
        
        global print
        def print(s, end=''):
            if end == '\r':
                self.cursor.select(QTextCursor.LineUnderCursor)
                self.cursor.removeSelectedText()
                self.console.insertPlainText(str(s))
            else:
                self.console.appendPlainText(str(s))
            self.console.repaint()
        
        self.console.appendPlainText("********** PyPO Console **********")
        # self.console.log()
        
        # global print
        # def print(s, end=''):
        #     #s += end
        #     if end == '\r':
        #         self.cursor.setPosition(QTextCursor.End)
        #         self.cursor.select(QTextCursor.LineUnderCursor)
        #         self.cursor.removeSelectedText()
        #         self.console.insertPlainText(s)
        #     else:
        #         self.console.appendPlainText(s)
        #     self.console.repaint()
        self.addToWindowGrid(self.console, self.GPConsole)
    
    def _mkElementsColumn(self):
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)

        self.refletorActions = [self.setTransromationForm, self.plotElement, self.removeElement]
        StmElements = []
        for e in self.stm.system.keys():
            StmElements.append(e)
        self.ElementsColumn = Accordion()

        scroll = QScrollArea()
        scroll.setWidget(self.ElementsColumn)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(300)
        self.addToWindowGrid(scroll, self.GPElementsColumn)

    def _setupPlotScreen(self):
        self.PlotScreen = QTabWidget()
        self.PlotScreen.setTabsClosable(True)
        self.PlotScreen.setTabShape(QTabWidget.Rounded)
        self.PlotScreen.tabCloseRequested.connect(self.closeTab)
        self.PlotScreen.setMaximumHeight(550)
        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def _formatVector(self, vector):
        return f"[{vector[0]}, {vector[1]}, {vector[2]}]"

    def addPlot(self, figure, label):
        self.PlotScreen.addTab(PlotScreen(figure, parent=self), label)
        self.PlotScreen.setCurrentIndex(self.PlotScreen.count()-1)


    def closeTab(self, i):
        self.PlotScreen.removeTab(i)

    def removeElement(self, element):
        self.stm.removeElement(element)
        
    def plotElement(self, surface):
        if self.stm.system:
            figure, _ = self.stm.plot3D(surface, show=False, save=False, ret=True)
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
            for key in self.stm.frames.keys():
                framelist.append(key)
        
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False, RTframes=framelist)
        
        else:
            figure = None
        self.addPlot(figure,"Ray Trace Frame %d" %(self.getRayPlotNr()))

    def saveSystemAction(self):
        self.setForm(fDataObj.saveSystemForm(), readAction=self.saveSystemCall)
    
    def loadSystemAction(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.loadSystemCall)
    
    def removeSystemAction(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.removeSystemCall)
    
    def saveSystemCall(self):
        saveDict = self.ParameterWid.read()
        self.stm.saveSystem(saveDict["name"]) 
    
    def loadSystemCall(self):
        loadDict = self.ParameterWid.read()
        self._mkElementsColumn()
        self.stm.loadSystem(loadDict["name"]) 
        self.refreshColumn(self.stm.system, "elements")
        self.refreshColumn(self.stm.frames, "frames")
        self.refreshColumn(self.stm.fields, "fields")
        self.refreshColumn(self.stm.currents, "currents")
   
    def removeSystemCall(self):
        removeDict = self.ParameterWid.read()
        shutil.rmtree(os.path.join(self.stm.savePathSystems, removeDict["name"]))

    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])

    def refreshColumn(self, columnDict, columnType):
        for key, item in columnDict.items():
            if columnType == "elements":
                self.ElementsColumn.reflectors.addWidget(ElementWidget(key, self.refletorActions))
            
            elif columnType == "frames":
                self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(key, 
                        [self.setPlotFrameFormOpt, self.stm.removeFrame, self.calcRMSfromFrame]))
            
            elif columnType == "fields":
                self.ElementsColumn.POFields.addWidget(FieldsWidget(key, [self.setPlotFieldFormOpt, self.stm.removeField]))
            
            elif columnType == "currents":
                self.ElementsColumn.POCurrents.addWidget(CurrentWidget(key, [self.setPlotFieldFormOpt, self.stm.removeCurrent]))

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
            "gridsize"  : np.array([501,501])
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
        # self.ParameterWid.setContentsMargins(5,5,5,5)
        scroll = QScrollArea()
        scroll.setWidget(self.ParameterWid)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll.border
        scroll.setWidgetResizable(True)
        scroll.setContentsMargins(0,0,0,0)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)
        self.addToWindowGrid(scroll,self.GPParameterForm)

    def setQuadricForm(self):
        self.setForm(fDataObj.makeQuadricSurfaceInp(), readAction=self.addQuadricAction)
    
    def setPlaneForm(self):
        self.setForm(fDataObj.makePlaneInp(), readAction=self.addPlaneAction)

    def addQuadricAction(self):
        try:
            elementDict = self.ParameterWid.read()
            if elementDict["type"] == "Parabola":
                self.stm.addParabola(elementDict)
            elif elementDict["type"] == "Hyperbola":
                self.stm.addHyperbola(elementDict)
            elif elementDict["type"] == "Ellipse":
                self.stm.addEllipse(elementDict)
            self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
        except InputReflError as e:
            self.console.appendPlainText("FormInput Incorrect:")
            self.console.appendPlainText(e.__str__())

    def addParabolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addParabola(elementDict) 
    
    def addHyperbolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addHyperbola(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))

    def addEllipseAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addEllipse(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
    
    def addPlaneAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addPlane(elementDict) 
        self.ElementsColumn.reflectors.addWidget(ElementWidget(elementDict["name"],self.refletorActions))
    
    def setTransromationForm(self, element):
        self.setForm(fDataObj.makeTransformationForm(element), self.applyTransformation)

    def applyTransformation(self, element):
        dd = self.ParameterWid.read()
        transformationType = dd["type"]
        vector = dd["vector"]

        if transformationType == "Translation":
            self.stm.translateGrids(dd["element"], vector)
            print(f'Translated {dd["element"]} by {self._formatVector(vector)} mm')
        elif transformationType == "Rotation":
            self.stm.rotateGrids(dd["element"], vector, cRot=dd["centerOfRotation"])
            print(f'Rotated {dd["element"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["centerOfRotation"])}')
        else:
            raise Exception("Transformation type incorrect")

    #NOTE Raytrace widgets
    def setInitFrameForm(self):
        self.setForm(fDataObj.initFrameInp(), readAction=self.addFrameAction)

    def setInitGaussianForm(self):
        self.setForm(fDataObj.initGaussianInp(self.stm.system), readAction=self.addGaussianAction)
    
    def setInitPSForm(self):
        self.setForm(fDataObj.initPSInp(self.stm.system), readAction=self.addPSAction)
    
    def addFrameAction(self):
        RTDict = self.ParameterWid.read()
        self.stm.createFrame(RTDict)
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(RTDict["name"],
                            [self.setPlotFrameFormOpt, self.stm.removeFrame, self.calcRMSfromFrame]))    
    
    def addGaussianAction(self):
        GDict = self.ParameterWid.read()
        self.stm.createGauss(GDict, GDict["surface"])
        self.ElementsColumn.POFields.addWidget(FieldsWidget(GDict["name"], [self.setPlotFieldFormOpt, self.stm.removeField]))
        self.ElementsColumn.POCurrents.addWidget(CurrentWidget(GDict["name"], [self.setPlotCurrentFormOpt, self.stm.removeCurrent]))
    
    def addPSAction(self):
        PSDict = self.ParameterWid.read()
        self.stm.generatePointSource(PSDict, PSDict["surface"])
        self.ElementsColumn.POFields.addWidget(FieldsWidget(PSDict["name"], [self.setPlotFieldFormOpt, self.stm.removeField]))
        self.ElementsColumn.POCurrents.addWidget(CurrentWidget(PSDict["name"], [self.setPlotCurrentFormOpt, self.stm.removeCurrent]))
    
    def setPlotFrameForm(self):
        self.setForm(fDataObj.plotFrameInp(self.stm.frames), readAction=self.addPlotFrameAction)
    
    def setPlotFrameFormOpt(self, frame):
        self.setForm(fDataObj.plotFrameOpt(frame), readAction=self.addPlotFrameAction)

    def setPlotFieldFormOpt(self, field):
        if self.stm.system[self.stm.fields[field].surf]["gmode"] == 2:
            self.setForm(fDataObj.plotFarField(field), readAction=self.addPlotFieldAction)
        else:
            self.setForm(fDataObj.plotField(field), readAction=self.addPlotFieldAction)
            
        
    def setPlotCurrentFormOpt(self, current):
        self.setForm(fDataObj.plotCurrentOpt(current), readAction=self.addPlotCurrentAction)
    
    def addPlotFrameAction(self):
        plotFrameDict = self.ParameterWid.read()
        fig = self.stm.plotRTframe(plotFrameDict["frame"], project=plotFrameDict["project"], returns=True)
        self.addPlot(fig, f'{plotFrameDict["frame"]} - {plotFrameDict["project"]}')

        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)

    def addPlotFieldAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], plotFieldDict["comp"], 
                                    ptype="field", project=plotFieldDict["project"], returns=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)
    
    def addPlotCurrentAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(self.stm.currents[plotFieldDict["field"]].surf, plotFieldDict["field"], 
                                    plotFieldDict["comp"], ptype="current", project=plotFieldDict["project"], returns=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotScreen, self.GPPlotScreen)
    
    def setPropRaysForm(self):
        self.setForm(fDataObj.propRaysInp(self.stm.frames, self.stm.system), self.addPropRaysAction)

    def addPropRaysAction(self): 
        propRaysDict = self.ParameterWid.read()
        self.stm.runRayTracer(propRaysDict["frame_in"], propRaysDict["frame_out"], 
                            propRaysDict["target"], propRaysDict["epsilon"], propRaysDict["nThreads"], 
                            propRaysDict["t0"], propRaysDict["device"], verbose=False)
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(propRaysDict["frame_out"], 
                                [self.setPlotFrameFormOpt, self.stm.removeFrame, self.calcRMSfromFrame]))
    
    def setPOInitForm(self):
        self.setForm(fDataObj.propPOInp(self.stm.currents, self.stm.system), self.addPropBeamAction)
    
    def setPOFFInitForm(self):
        self.setForm(fDataObj.propPOFFInp(self.stm.currents, self.stm.system), self.addPropBeamAction)
    
    def setTaperEffsForm(self):
        self.setForm(fDataObj.calcTaperEff(self.stm.fields, self.stm.system), self.calcTaperAction)
    
    def setSpillEffsForm(self):
        self.setForm(fDataObj.calcSpillEff(self.stm.fields, self.stm.system), self.calcSpillAction)

    def setXpolEffsForm(self):
        self.setForm(fDataObj.calcXpolEff(self.stm.fields, self.stm.system), self.calcXpolAction)

    def calcTaperAction(self):
        TaperDict = self.ParameterWid.read()
        eff_taper = self.stm.calcTaper(TaperDict["f_name"], TaperDict["comp"])
        print(f'Taper efficiency of {TaperDict["f_name"]}, component {TaperDict["comp"]} = {eff_taper}\n')
    
    def calcSpillAction(self):
        SpillDict = self.ParameterWid.read()

        aperDict = {
                "center"    : SpillDict["center"],
                "inner"      : SpillDict["inner"],
                "outer"      : SpillDict["outer"]
                }

        eff_spill = self.stm.calcSpillover(SpillDict["f_name"], SpillDict["comp"], aperDict)
        print(f'Spillover efficiency of {SpillDict["f_name"]}, component {SpillDict["comp"]} = {eff_spill}\n')
    
    def calcXpolAction(self):
        XpolDict = self.ParameterWid.read()
        eff_Xpol = self.stm.calcXpol(XpolDict["f_name"], XpolDict["co_comp"], XpolDict["cr_comp"])
        print(f'X-pol efficiency of {XpolDict["f_name"]}, co-component {XpolDict["co_comp"]} and X-component {XpolDict["cr_comp"]} = {eff_Xpol}\n')

    def addPropBeamAction(self):
        propBeamDict = self.ParameterWid.read()
        self.stm.runPO(propBeamDict)

        if propBeamDict["mode"] == "JM":
            self.ElementsColumn.POCurrents.addWidget(CurrentWidget(propBeamDict["name_JM"], [self.setPlotCurrentFormOpt, self.stm.removeCurrent]))
        
        elif propBeamDict["mode"] == "EH" or propBeamDict["mode"] == "FF":
            self.ElementsColumn.POFields.addWidget(FieldsWidget(propBeamDict["name_EH"], [self.setPlotFieldFormOpt, self.stm.removeField]))
        
        elif propBeamDict["mode"] == "JMEH":
            self.ElementsColumn.POCurrents.addWidget(CurrentWidget(propBeamDict["name_JM"], [self.setPlotCurrentFormOpt, self.stm.removeCurrent]))
            self.ElementsColumn.POFields.addWidget(FieldsWidget(propBeamDict["name_EH"], [self.setPlotFieldFormOpt, self.stm.removeField]))
        
        elif propBeamDict["mode"] == "EHP":
            self.ElementsColumn.POFields.addWidget(CurrentWidget(propBeamDict["name_EH"], [self.setPlotCurrentFormOpt, self.stm.removeCurrent]))
            self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(propBeamDict["name_P"], 
                                [self.setPlotFrameFormOpt, self.stm.removeFrame, self.calcRMSfromFrame]))
    #END NOTE
    
    def calcRMSfromFrame(self, frame):
        rms = self.stm.calcSpotRMS(frame)
        print(f"RMS value of {frame} = {rms} mm\n")

class PyPOMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.mainWid = MainWidget()
        self.mainWid.setContentsMargins(0,0,0,0)
        self.setContentsMargins(0,0,0,0)
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
        PhysOptMenu = menuBar.addMenu("Physical-optics")

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
        
        saveSystem = QAction("Save system", self)
        saveSystem.triggered.connect(self.mainWid.saveSystemAction)
        SystemsMenu.addAction(saveSystem)

        loadSystem = QAction("Load system", self)
        loadSystem.triggered.connect(self.mainWid.loadSystemAction)
        SystemsMenu.addAction(loadSystem)
        
        removeSystem = QAction("Remove system", self)
        removeSystem.triggered.connect(self.mainWid.removeSystemAction)
        SystemsMenu.addAction(removeSystem)
        
        # NOTE Raytrace actions
        makeFrame = RaytraceMenu.addMenu("Make frame")
        initFrameAction = QAction("Initialize", self)
        initFrameAction.setStatusTip("Initialize ray-trace frame from input form")
        initFrameAction.triggered.connect(self.mainWid.setInitFrameForm)
        makeFrame.addAction(initFrameAction)
        
        poyntingFrameAction = QAction("Poynting", self)


        # Propagate rays
        propRaysAction = QAction("Propagate rays", self)
        propRaysAction.triggered.connect(self.mainWid.setPropRaysForm)
        RaytraceMenu.addAction(propRaysAction)

        # PO actions
        makeBeam = PhysOptMenu.addMenu("Initialize beam")
        initPointAction = QAction("Point source", self)
        initPointAction.triggered.connect(self.mainWid.setInitPSForm)
        makeBeam.addAction(initPointAction)
    
        initGaussAction = QAction("Gaussian beam", self)
        initGaussAction.triggered.connect(self.mainWid.setInitGaussianForm)
        makeBeam.addAction(initGaussAction)

        propBeam = PhysOptMenu.addMenu("Propagate beam") 
        initPropSurfAction = QAction("To surface", self)
        initPropSurfAction.triggered.connect(self.mainWid.setPOInitForm)
        propBeam.addAction(initPropSurfAction)
        
        initPropFFAction = QAction("To far-field", self)
        initPropFFAction.triggered.connect(self.mainWid.setPOFFInitForm)
        propBeam.addAction(initPropFFAction)

        calcEffs = PhysOptMenu.addMenu("Efficiencies")
        calcSpillEffsAction = QAction("Spillover", self)
        calcSpillEffsAction.triggered.connect(self.mainWid.setSpillEffsForm)
        calcEffs.addAction(calcSpillEffsAction)
        
        calcTaperEffsAction = QAction("Taper", self)
        calcTaperEffsAction.triggered.connect(self.mainWid.setTaperEffsForm)
        calcEffs.addAction(calcTaperEffsAction)
        
        calcXpolEffsAction = QAction("X-pol", self)
        calcXpolEffsAction.triggered.connect(self.mainWid.setXpolEffsForm)
        calcEffs.addAction(calcXpolEffsAction)

        # END NOTE
if __name__ == "__main__":

    print("lala")
    app = QApplication(sys.argv)
    win = PyPOMainWindow(parent=None)
    # def print(s):
    #     cons.appendPlainText(s)
    win.show()
    app.exec_()
