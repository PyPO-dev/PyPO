import os
import sys
import shutil
import asyncio

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt

from src.GUI.ParameterForms import formGenerator
from src.GUI.ParameterForms.InputDescription import InputDescription
from src.GUI.utils import inType
import src.GUI.ParameterForms.formDataObjects as fDataObj
from src.GUI.PlotScreen import PlotScreen
from src.GUI.TransformationWidget import TransformationWidget
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ReflectorWidget, FrameWidget, FieldsWidget, CurrentWidget, SFieldsWidget, SymDialog
from src.GUI.Console import ConsoleGenerator

from src.GUI.Console import print
import numpy as np
from src.PyPO.Checks import InputReflError, InputRTError

sys.path.append('../')
sys.path.append('../../')
import src.PyPO.System as st
import src.PyPO.Threadmgr as TManager




##
# @file 
# defines classes PyPOMainWindow and MainWidget
# PyPOMainWindow is responsible for setting up the window and toolbars
# MainWidget is responsble for all gui functionalities
#
class MainWidget(QWidget):
    ##
    # Constructot. Configures the layout and initializes the underlying systemmlala
    # @see System
    # 
    # 

    def __init__(self, parent=None):
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("PyPO")

        # init System
        self.stm = st.System(redirect=print, context="G")
        self.pyprint = print


        # GridParameters
        self.GPElementsColumn = [0, 0, 2, 1]
        self.GPParameterForm  = [0, 1, 2, 1]
        self.GPPlotScreen     = [0, 2, 1, 1]
        self.GPConsole        = [1, 2, 1, 1]

        
        # init layout
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0,0,0,0)
        # self.grid.setMargin(0)
        self.grid.setSpacing(0)

        self._mkElementsColumn()
        self._mkPlotScreen()
        self._mkConsole()
        self.setLayout(self.grid)


        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE

    ### Gui setup functions
    ##
    # @guiSetup
    # Adds a widget to the layout of PyPOMainWidget
    def addToWindowGrid(self, widget, param):
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])


    ##  
    # @guiSetup
    # Configures the console widget
    # 
    def _mkConsole(self):
        self.console = ConsoleGenerator.get()
        self.addToWindowGrid(self.console, self.GPConsole)
        self.cursor = QTextCursor(self.console.document())
        
        global print ##TODO: Remove print redefinitions
        def print(s, end=''):
            if end == '\r':
                self.cursor.select(QTextCursor.LineUnderCursor)
                self.cursor.removeSelectedText()
                self.console.insertPlainText(str(s))
            else:
                self.console.appendPlainText(str(s))
            self.console.repaint()
        
        self.console.appendPlainText("********** PyPO Console **********")
        self.addToWindowGrid(self.console, self.GPConsole)
    
    ##
    # @guiSetup
    # constructs the elements column on the left side of the screen
    # 
    def _mkElementsColumn(self):
        # delete if exists
        if hasattr(self, "ElementsColumn"):
            self.ElementsColumn.setParent(None)
        # rebuild 
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

    ##
    # @guiSetup
    # Constructs the tab widget which will later hold plots
    def _mkPlotScreen(self):
        self.PlotWidget = QTabWidget()
        self.PlotWidget.setTabsClosable(True)
        self.PlotWidget.setTabShape(QTabWidget.Rounded)
        self.PlotWidget.tabCloseRequested.connect(self.closePlotTab)
        self.PlotWidget.setMaximumHeight(550)
        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)

    ##
    # @guiSetup
    # Generates a form widget
    # 
    # @param formData List of InputDescription objects
    # @param readAction Function to be called when forms ok-button is clicked
    # 
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
    
    ## 
    # removes the form ParameterWid if exists 
    def removeForm(self):
        if hasattr(self, "ParameterWid"):
            self.ParameterWid.setParent(None)
    
    ##
    # TODO: Remove this function from here
    @staticmethod
    def _formatVector(vector):
        return f"[{vector[0]}, {vector[1]}, {vector[2]}]"
    
    ### Functionalities: Plotting

    ##
    # Constructs a PlotScreen and adds it to the plotWidget along with a label for the tab
    def addPlot(self, figure, label):
        self.PlotWidget.addTab(PlotScreen(figure, parent=self), label)
        self.PlotWidget.setCurrentIndex(self.PlotWidget.count()-1)

    ##
    # plots a single element from the System
    # 
    # @param surface str Name of the surface in system
    # 
    def plotElement(self, surface):
        if self.stm.system:
            figure, _ = self.stm.plot3D(surface, show=False, save=False, ret=True)
        else :
            figure = None
        self.addPlot(figure, surface)

    ##
    # Generate a snapshot form.
    def snapActionForm(self, element):
        self.setForm(fDataObj.snapForm(element, list(self.stm.system[element]["snapshots"].keys()), "element"), readAction=self.snapAction)
 
    ##
    # Generate a snapshot form for ray-trace frame.
    def snapFrameActionForm(self, frame):
        self.setForm(fDataObj.snapForm(frame, list(self.stm.frames.snapshots.keys()), "frame"), readAction=self.snapAction)
    
    ##
    # Take, revert or delete a snapshot
    # 
    # @param element Name of the surface in system
    # 
    def snapAction(self):
        snapDict = self.ParameterWid.read()
        if snapDict["options"] == "Take":
            self.stm.snapObj(snapDict["name"], snapDict["snap_name"], snapDict["obj"])
        
        elif snapDict["options"] == "Revert":
            self.stm.revertToSnap(snapDict["name"], snapDict["snap_name"], snapDict["obj"])
        
        elif snapDict["options"] == "Delete":
            self.stm.deleteSnap(snapDict["name"], snapDict["snap_name"], snapDict["obj"])
    
    ##
    # plots all elements of the system in one plot
    def plotSystem(self):
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False)
        else :
            figure = None
        self.addPlot(figure, "System Plot %d" %self.getSysPlotNr())
        
    ##
    # removes a plot from the tabWidget
    # @param i Index of the plot to be removed
    def closePlotTab(self, i):
        self.PlotWidget.removeTab(i)
    ### Functionalities: Naming 
    ##
    # Gets the plot number and increments it 
    # @return The incremented plot number 
    def getSysPlotNr(self):
        if not hasattr(self, "sysPlotNr"):
            self.sysPlotNr = 0
        self.sysPlotNr+=1
        return self.sysPlotNr

    ##
    # Gets the frame plot number and increments it
    # @return The incremented frame plot number 
    def getRayPlotNr(self):
        if not hasattr(self, "rayPlotNr"):
            self.rayPlotNr = 0
        self.rayPlotNr+=1
        return self.rayPlotNr

    ##
    # plots all elements of the system including ray traces in one plot
    def plotSystemWithRaytrace(self):
        framelist = []

        if self.stm.frames:
            for key in self.stm.frames.keys():
                framelist.append(key)
        
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False, RTframes=framelist)
        
        else:
            figure = None
        self.addPlot(figure,"Ray Trace Frame %d" %(self.getRayPlotNr()))

    ### Functionalities: Systems 


    ##
    # opens a form that allows user to save the System
    def saveSystemForm(self):
        self.setForm(fDataObj.saveSystemForm(), readAction=self.saveSystemAction)
    
    ##
    # Saves the current system state under the name given in form
    def saveSystemAction(self):
        saveDict = self.ParameterWid.read()
        self.stm.saveSystem(saveDict["name"]) 
    
    
    ##
    # opens a form that allows user to delete a saved System
    def deleteSavedSystemForm(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.deleteSavedSystemAction)

    ##
    # Deletes system selected in form
    def deleteSavedSystemAction(self):
        removeDict = self.ParameterWid.read()
        shutil.rmtree(os.path.join(self.stm.savePathSystems, removeDict["name"]))

    ##
    # opens a form that allows user to load a saved System
    def loadSystemForm(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fDataObj.loadSystemForm(systemList), readAction=self.loadSystemAction)
    
    ##
    # Loads system selected in from form
    def loadSystemAction(self):
        loadDict = self.ParameterWid.read()
        self._mkElementsColumn()
        self.stm.loadSystem(loadDict["name"]) 
        self.refreshColumn(self.stm.system, "elements")
        self.refreshColumn(self.stm.frames, "frames")
        self.refreshColumn(self.stm.fields, "fields")
        self.refreshColumn(self.stm.currents, "currents")
        self.refreshColumn(self.stm.scalarfields, "scalarfields")


    ##
    # removes an element from the system
    # @param element Name of the element in the system
    def removeElement(self, element):
        print(f"removed: {element}") # TODO remove print redirection
        self.stm.removeElement(element)
    
    ##
    # TODO: @Maikel Rename this function and evaluate its nessecity
    def refreshColumn(self, columnDict, columnType):
        for key, item in columnDict.items():
            if columnType == "elements":
                self.addReflectorWidget(key)
            
            elif columnType == "frames":
                self.addFrameWidget(key)
            
            elif columnType == "fields":
                self.addFieldWidget(key)
            
            elif columnType == "currents":
                self.addCurrentWidget(key)

            elif columnType == "scalarfields":
                self.addSFieldWidget(key)

    ### Functionalities: Adding Elements in gui
    # TODO:doc
    def addReflectorWidget(self, name):
        self.ElementsColumn.reflectors.addWidget(ReflectorWidget(name, self.removeElement, self.transformSingleForm, self.plotElement, self.snapActionForm)) 

    def addFrameWidget(self, name):
        self.ElementsColumn.RayTraceFrames.addWidget(FrameWidget(name, self.stm.removeFrame, 
                                                    self.transformFrameForm, self.plotFrameForm,  
                                                    self.calcRMSfromFrame, self.snapFrameActionForm))

    def addFieldWidget(self, name):
        self.ElementsColumn.POFields.addWidget(FieldsWidget(name,self.stm.removeField, self.plotFieldForm))

    def addCurrentWidget(self, name):
        self.ElementsColumn.POCurrents.addWidget(CurrentWidget(name, self.stm.removeCurrent, self.plotFieldForm))

    def addSFieldWidget(self, name):
        self.ElementsColumn.SPOFields.addWidget(SFieldsWidget(name,self.stm.removeScalarField, self.plotSFieldForm))


    ### Functionalities: Adding Elements 

    ##
    # Shows form to add a plane
    def addPlaneForm(self):
        self.setForm(fDataObj.makePlaneInp(), readAction=self.addPlaneAction)

    ##
    # Reads form and adds plane to System
    def addPlaneAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addPlane(elementDict) 
        self.addReflectorWidget(elementDict["name"])
    
    ##
    # Shows from to add a quadric surface
    def addQuadricForm(self):
        self.setForm(fDataObj.makeQuadricSurfaceInp(), readAction=self.addQuadricAction)
    
    ##
    # Reads quadric form, evaluates surface type and calls corresponding add____Action
    def addQuadricAction(self):
        try:
            elementDict = self.ParameterWid.read()
            if elementDict["type"] == "Parabola":
                self.stm.addParabola(elementDict)
            elif elementDict["type"] == "Hyperbola":
                self.stm.addHyperbola(elementDict)
            elif elementDict["type"] == "Ellipse":
                self.stm.addEllipse(elementDict)

            self.addReflectorWidget(elementDict["name"])
        except InputReflError as e: #TODO: Does this errorCatching work?
            self.console.appendPlainText("FormInput Incorrect:")
            self.console.appendPlainText(e.__str__())

    ##
    # Reads form and adds parabola to System
    def addParabolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addParabola(elementDict) 
        self.addReflectorWidget(elementDict["name"])
    
    ##
    # Reads form and adds hyperbola to System
    def addHyperbolaAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addHyperbola(elementDict) 
        self.addReflectorWidget(elementDict["name"])

    ##
    # Reads form and adds ellipse to System
    def addEllipseAction(self):
        elementDict = self.ParameterWid.read()
        self.stm.addEllipse(elementDict) 
        self.addReflectorWidget(elementDict["name"])
    
    ### Functionalities: Transforming Elements 

    ##
    # Shows single element transformation form
    def transformSingleForm(self, element):
        self.setForm(fDataObj.makeTransformationForm(element), self.transformAction)
    
    ##
    # Shows single element transformation form
    def transformFrameForm(self, frame):
        self.setForm(fDataObj.makeTransformationForm(frame, obj="frame"), self.transformFrameAction)
    
    ##
    # Applies single element transformation
    def transformAction(self, element):
        dd = self.ParameterWid.read()
        transformationType = dd["type"]
        vector = dd["vector"]

        if transformationType == "Translation":
            self.stm.translateGrids(dd["element"], vector, mode=dd["mode"].lower())
            print(f'Translated {dd["element"]} by {self._formatVector(vector)} mm')
        elif transformationType == "Rotation":
            self.stm.rotateGrids(dd["element"], vector, pivot=dd["pivot"], mode=dd["mode"].lower())
            print(f'Rotated {dd["element"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["pivot"])}')
        else:
            raise Exception("Transformation type incorrect")

    ##
    # Applies single frame transformation
    def transformFrameAction(self, frame):
        dd = self.ParameterWid.read()
        transformationType = dd["type"]
        vector = dd["vector"]

        if transformationType == "Translation":
            self.stm.translateGrids(dd["frame"], vector, mode=dd["mode"].lower(), obj="frame")
            print(f'Translated {dd["frame"]} by {self._formatVector(vector)} mm')
        elif transformationType == "Rotation":
            self.stm.rotateGrids(dd["frame"], vector, pivot=dd["pivot"], mode=dd["mode"].lower(), obj="frame")
            print(f'Rotated {dd["frame"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["pivot"])}')
        else:
            raise Exception("Transformation type incorrect")
    
    ##
    # Shows multiple element transformation form
    def transformationMultipleForm(self):
        movableElements = []
        for key, elem in self.stm.system.items():
            if elem["gmode"] != 2:
                movableElements.append(key)

        self.setForm(
            [InputDescription(inType.elementSelector, "elements", options=movableElements)]+
            fDataObj.makeTransformationElementsForm(self.stm.system.keys()), self.transformationMultipleAction
            )

    ##
    # Applies multiple element transformation
    def transformationMultipleAction(self):
        transfDict = self.ParameterWid.read()

        if transfDict["type"] == "Translation":
            self.stm.translateGrids(transfDict["elements"], transfDict["vector"])
            print(f'Translated {transfDict["elements"]} by {self._formatVector(transfDict["vector"])} mm')

        if transfDict["type"] == "Rotation":
            self.stm.rotateGrids(transfDict["elements"], transfDict["vector"], transfDict["pivot"])
            print(f'Translated {transfDict["elements"]} by {self._formatVector(transfDict["vector"])} mm')
    
    ### Functionalities: TRFrames 
    #NOTE Raytrace widgets

    ##
    # Shows tube frame form
    def initTubeFrameForm(self):
        self.setForm(fDataObj.initTubeFrameInp(), readAction=self.initTubeFrameAction)
    
    ##
    # Reads form and adds a tube frame to system
    def initTubeFrameAction(self):
        RTDict = self.ParameterWid.read()
        self.stm.createTubeFrame(RTDict)
        self.addFrameWidget(RTDict["name"])
    
    ##
    # Shows form to initialize gaussian frame 
    def initGaussianFrameForm(self):
        self.setForm(fDataObj.initGaussianFrameInp(), readAction=self.initGaussianFrameAction)
    

    ##
    # Reads form and adds a gaussian frame to system
    def initGaussianFrameAction(self):
        GRTDict = self.ParameterWid.read()

        if not "seed" in GRTDict.keys():
            GRTDict["seed"] = -1

        self.stm.createGRTFrame(GRTDict)
        self.addFrameWidget(GRTDict["name"])
        
    ##
    # Shows form to propagate rays
    def setPropRaysForm(self):
        self.setForm(fDataObj.propRaysInp(self.stm.frames, self.stm.system), self.addPropRaysAction)

    ##
    # Reads form and popagates rays
    def addPropRaysAction(self): 
        propRaysDict = self.ParameterWid.read()
        self.stm.runRayTracer(propRaysDict)
        self.addFrameWidget(propRaysDict["fr_out"])

    ##
    # Shows form to plot preselected a frame
    # 
    # @param frame Frame to plot
    def plotFrameForm(self, frame):
        self.setForm(fDataObj.plotFrameOpt(frame), readAction=self.addPlotFrameAction)

    ##
    # Reads form and plots frame
    def addPlotFrameAction(self):
        plotFrameDict = self.ParameterWid.read()
        fig = self.stm.plotRTframe(plotFrameDict["frame"], project=plotFrameDict["project"], ret=True)
        self.addPlot(fig, f'{plotFrameDict["frame"]} - {plotFrameDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)

    ### Functionalities: PO 
    ##
    # Shows form to initialize gaussian beam 
    def initGaussBeamForm(self):
        self.setForm(fDataObj.initGaussianInp(self.stm.system), readAction=self.initGaussBeamAction)

    ##
    # Reads form and adds a vectorial gaussian beam to system
    def initGaussBeamAction(self):
        GDict = self.ParameterWid.read()
        self.stm.createGaussian(GDict, GDict["surface"])
        self.addFieldWidget(GDict["name"])
        self.addCurrentWidget(GDict["name"])
    
    
    ##
    # Shows form to initialize scalar gaussian beam TODO: klopt dit 
    def initSGaussBeamForm(self):
        self.setForm(fDataObj.initSGaussianInp(self.stm.system), readAction=self.initSGaussBeamAction)
    
    ##
    # Reads form and adds a scalar gaussian beam to system
    def initSGaussBeamAction(self):
        GDict = self.ParameterWid.read()
        self.stm.createScalarGaussian(GDict, GDict["surface"])
        self.addSFieldWidget(GDict["name"])

    ##
    # Shows form to initialize a physical optics propagation
    def initPSBeamForm(self):
        self.setForm(fDataObj.initPSInp(self.stm.system), readAction=self.initPSBeamAction)
    
    
    ##
    # Reads form and adds a vectorial point source beam to system
    def initPSBeamAction(self):
        PSDict = self.ParameterWid.read()
        self.stm.generatePointSource(PSDict, PSDict["surface"])
        self.addFieldWidget(PSDict["name"])
        self.addCurrentWidget(PSDict["name"])
    
    ##
    # Shows form to initialize a scalar point source beam
    def initSPSBeamForm(self):
        self.setForm(fDataObj.initSPSInp(self.stm.system), readAction=self.initSPSBeamAction)
    

    ##
    # Reads form and adds a scalar point source beam to system
    def initSPSBeamAction(self):
        SPSDict = self.ParameterWid.read()
        self.stm.generatePointSourceScalar(SPSDict, SPSDict["surface"])
        self.addSFieldWidget(SPSDict["name"])
    


    ##
    # Shows form to plot field
    #
    # @param field Field to plot
    def plotFieldForm(self, field):
        if self.stm.system[self.stm.fields[field].surf]["gmode"] == 2:
            self.setForm(fDataObj.plotFarField(field), readAction=self.plotFieldAction)
        else:
            self.setForm(fDataObj.plotField(field), readAction=self.plotFieldAction)

    ##
    # Reads form and plots field
    def plotFieldAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], plotFieldDict["comp"], 
                                    project=plotFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
                
    ##
    # TODO: whats the difference with above? Perhase we should rename function
    #
    # @param field Field to plot
    def plotSFieldForm(self, field):
        self.setForm(fDataObj.plotSField(field, self.stm.system[self.stm.scalarfields[field].surf]["gmode"]), readAction=self.plotSFieldAction)

    ##
    # Reads form and plots scalar field
    def plotSFieldAction(self):
        plotSFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotSFieldDict["field"], 
                                    project=plotSFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotSFieldDict["field"]} - {plotSFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
    
    
    ## 
    # Shows form to plot current
    #
    # @param current Current to plot
    def plotCurrentForm(self, current):
        self.setForm(fDataObj.plotCurrentOpt(current), readAction=self.plotCurrentAction)
    

    ##
    # Reads form and plots current
    def plotCurrentAction(self):
        plotFieldDict = self.ParameterWid.read()
        fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], 
                                    plotFieldDict["comp"], project=plotFieldDict["project"], ret=True)
        self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)
    
    
    
    ##
    # Shows form to propagate physical optics beam to surface 
    def propPOForm(self):
        self.setForm(fDataObj.propPOInp(self.stm.currents, self.stm.scalarfields, self.stm.system), self.propPOAction)
    
    ##
    # Shows form to propagate physical optics beam far field 
    def propPOFFForm(self):
        self.setForm(fDataObj.propPOFFInp(self.stm.currents, self.stm.system), self.propPOAction)
    
    ##
    # Reads form propagates beam, runs calculation on another thread
    def propPOAction(self):
        propBeamDict = self.ParameterWid.read()
      
        if propBeamDict["mode"] == "scalar":
            subStr = "scalar field"
        else:
            subStr = propBeamDict["mode"]

        dialStr = f"Calculating {subStr} on {propBeamDict['t_name']}..."

        dial = SymDialog(dialStr)

        self.mgr = TManager.Manager("G", callback=dial.accept)
        t = self.mgr.new_gthread(target=self.stm.runPO, args=(propBeamDict,), calc_type=propBeamDict["mode"])
        
        dial.setThread(t)

        if dial.exec_():
            if propBeamDict["mode"] == "JM":
                self.addCurrentWidget(propBeamDict["name_JM"])
        
            elif propBeamDict["mode"] == "EH" or propBeamDict["mode"] == "FF":
                self.addFieldWidget(propBeamDict["name_EH"])
        
            elif propBeamDict["mode"] == "JMEH":
                self.addCurrentWidget(propBeamDict["name_JM"])
                self.addFieldWidget(propBeamDict["name_EH"])
        
            elif propBeamDict["mode"] == "EHP":
                self.addFieldWidget(propBeamDict["name_EH"])
                self.addFrameWidget(propBeamDict["name_P"])
    
            elif propBeamDict["mode"] == "scalar":
                self.addSFieldWidget(propBeamDict["name_field"])

    #TODO Unite efficiencies
    ##
    # Shows form to calculate taper efficientie
    def setTaperEffsForm(self):
        self.setForm(fDataObj.calcTaperEff(self.stm.fields, self.stm.system), self.calcTaperAction)
    
    ##
    # Shows form to calculate spillover efficientie
    def setSpillEffsForm(self):
        self.setForm(fDataObj.calcSpillEff(self.stm.fields, self.stm.system), self.calcSpillAction)

    ##
    # Shows form to calculate x-pol efficientie TODO: x-pol
    def setXpolEffsForm(self):
        self.setForm(fDataObj.calcXpolEff(self.stm.fields, self.stm.system), self.calcXpolAction)

    ##
    # Shows form to calculate main beam efficientie
    def setMBEffsForm(self):
        self.setForm(fDataObj.calcMBEff(self.stm.fields, self.stm.system), self.calcMBAction)
    
    ##
    # Reads form and calculates taper efficientie
    def calcTaperAction(self):
        TaperDict = self.ParameterWid.read()
        eff_taper = self.stm.calcTaper(TaperDict["f_name"], TaperDict["comp"])
        print(f'Taper efficiency of {TaperDict["f_name"]}, component {TaperDict["comp"]} = {eff_taper}\n')
    
    ##
    # Reads form and calculates spillover efficientie
    def calcSpillAction(self):
        SpillDict = self.ParameterWid.read()

        aperDict = {
                "center"    : SpillDict["center"],
                "inner"      : SpillDict["inner"],
                "outer"      : SpillDict["outer"]
                }

        eff_spill = self.stm.calcSpillover(SpillDict["f_name"], SpillDict["comp"], aperDict)
        print(f'Spillover efficiency of {SpillDict["f_name"]}, component {SpillDict["comp"]} = {eff_spill}\n')
    
    ##
    # Reads form and calculates x-pol efficientie TODO: x-pol?
    def calcXpolAction(self):
        XpolDict = self.ParameterWid.read()
        eff_Xpol = self.stm.calcXpol(XpolDict["f_name"], XpolDict["co_comp"], XpolDict["cr_comp"])
        print(f'X-pol efficiency of {XpolDict["f_name"]}, co-component {XpolDict["co_comp"]} and X-component {XpolDict["cr_comp"]} = {eff_Xpol}\n')

    ##
    # Reads form and calculates main beam efficientie
    def calcMBAction(self):
        MBDict = self.ParameterWid.read()
        eff_mb = self.stm.calcMainBeam(MBDict["f_name"], MBDict["comp"], MBDict["thres"], MBDict["mode"])
        print(f'Main beam efficiency of {MBDict["f_name"]}, component {MBDict["comp"]} = {eff_mb}\n')
        self.addSFieldWidget(f"fitGauss_{MBDict['f_name']}")

    
    #END NOTE
    
    ##
    # calculates root mean square of a frame
    def calcRMSfromFrame(self, frame):
        rms = self.stm.calcSpotRMS(frame)
        print(f"RMS value of {frame} = {rms} mm\n")

    def setFocusFindForm(self):
        self.setForm(fDataObj.focusFind(list(self.stm.frames.keys())), self.findFocusAction)

    def findFocusAction(self):
        print(self.ParameterWid.read())
        findFocusDict = self.ParameterWid.read()
        focus = self.stm.findRTfocus(findFocusDict["name_frame"], verbose=True) 
        print(f"Focus of {findFocusDict['name_frame']} = {focus}\n")

class PyPOMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.mainWid = MainWidget()
        self.mainWid.setContentsMargins(0,0,0,0)
        # self.setStyleSheet("background:red")
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

        ElementsMenu    = menuBar.addMenu("Elements")
        SystemsMenu     = menuBar.addMenu("Systems")
        RaytraceMenu    = menuBar.addMenu("Ray-tracer")
        PhysOptMenu     = menuBar.addMenu("Physical-optics")
        ToolsMenu       = menuBar.addMenu("Tools")

        # ### Generate test parabola
        # AddTestParabola = QAction('Add Test Parabola', self)
        # AddTestParabola.setShortcut('Ctrl+Shift+P')
        # AddTestParabola.setStatusTip('Generates a Parabolic reflector and plots it')
        # AddTestParabola.triggered.connect(self.mainWid.addExampleParabola)
        # ElementsMenu.addAction(AddTestParabola)

        # ### Generate test hyperbola
        # AddTestHyperbola = QAction('Add Test Hyperbola', self)
        # AddTestHyperbola.setShortcut('Ctrl+Shift+H')
        # AddTestHyperbola.setStatusTip('Generates a Parabolic reflector and plots it')
        # AddTestHyperbola.triggered.connect(self.mainWid.addExampleHyperbola)
        # ElementsMenu.addAction(AddTestHyperbola)

        ### Add Element
        reflectorSelector = ElementsMenu.addMenu("Reflector")
        ### Planar Surface
        planeAction = QAction("Plane", self)
        planeAction.setShortcut("Ctrl+L")
        planeAction.setStatusTip("Add a planar element.")
        planeAction.triggered.connect(self.mainWid.addPlaneForm)
        reflectorSelector.addAction(planeAction)
        
        ### Quadric Surface
        hyperbolaAction = QAction('Quadric surface', self)
        hyperbolaAction.setShortcut('Ctrl+Q')
        hyperbolaAction.setStatusTip("Add a paraboloid, hyperboloid or ellipsoid element.")
        hyperbolaAction.triggered.connect(self.mainWid.addQuadricForm)
        reflectorSelector.addAction(hyperbolaAction)

        transformElementsAction = QAction("Transform elements", self)
        transformElementsAction.setStatusTip("Transform a group of elements.")
        transformElementsAction.triggered.connect(self.mainWid.transformationMultipleForm)
        ElementsMenu.addAction(transformElementsAction)
        

    ### System actions
        # newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid.addSystemAction)
        # SystemsMenu.addAction(newSystem)

        plotSystem = QAction("Plot System", self)
        plotSystem.setStatusTip("Plot all elements in the current system.")
        plotSystem.triggered.connect(self.mainWid.plotSystem)
        SystemsMenu.addAction(plotSystem)

        plotRaytrace = QAction("Plot ray-trace", self)
        plotSystem.setStatusTip("Plot all elements in the current system, including ray-traces.")
        plotRaytrace.triggered.connect(self.mainWid.plotSystemWithRaytrace)
        SystemsMenu.addAction(plotRaytrace)
        
        saveSystem = QAction("Save system", self)
        saveSystem.setStatusTip("Save the current system to disk.")
        saveSystem.triggered.connect(self.mainWid.saveSystemForm)
        SystemsMenu.addAction(saveSystem)

        loadSystem = QAction("Load system", self)
        loadSystem.setStatusTip("Load a saved system from disk.")
        loadSystem.triggered.connect(self.mainWid.loadSystemForm)
        SystemsMenu.addAction(loadSystem)
        
        removeSystem = QAction("Remove system", self)
        removeSystem.setStatusTip("Remove a saved system from disk.")
        removeSystem.triggered.connect(self.mainWid.deleteSavedSystemForm)
        SystemsMenu.addAction(removeSystem)
        
        # NOTE Raytrace actions
        makeFrame = RaytraceMenu.addMenu("Make frame")
        initTubeFrameAction = QAction("Tube", self)
        initTubeFrameAction.setStatusTip("Initialize ray-trace tube from input form")
        initTubeFrameAction.triggered.connect(self.mainWid.initTubeFrameForm)
        makeFrame.addAction(initTubeFrameAction)

        initGaussianFrameAction = QAction("Gaussian", self)
        initGaussianFrameAction.setStatusTip("Initialize ray-trace Gaussian from input form")
        initGaussianFrameAction.triggered.connect(self.mainWid.initGaussianFrameForm)
        makeFrame.addAction(initGaussianFrameAction)
        
        # Propagate rays
        propRaysAction = QAction("Propagate rays", self)
        propRaysAction.setStatusTip("Propagate a frame of rays to a target surface")
        propRaysAction.triggered.connect(self.mainWid.setPropRaysForm)
        RaytraceMenu.addAction(propRaysAction)
        
        # PO actions
        makeBeam = PhysOptMenu.addMenu("Initialize beam")
        makeBeamPS = makeBeam.addMenu("Point source")
        initPointVecAction = QAction("Vectorial", self)
        initPointVecAction.setStatusTip("Initialize a vectorial point source.")
        initPointVecAction.triggered.connect(self.mainWid.initPSBeamForm)
        makeBeamPS.addAction(initPointVecAction)
        
        initPointScalAction = QAction("Scalar", self)
        initPointScalAction.setStatusTip("Initialize a scalar point source.")
        initPointScalAction.triggered.connect(self.mainWid.initSPSBeamForm)
        makeBeamPS.addAction(initPointScalAction)
    
        makeBeamG = makeBeam.addMenu("Gaussian beam")
        initGaussVecAction = QAction("Vectorial", self)#TODO Vectorial?
        initGaussVecAction.setStatusTip("Initialize a vectorial Gaussian beam.")
        initGaussVecAction.triggered.connect(self.mainWid.initGaussBeamForm)
        makeBeamG.addAction(initGaussVecAction)
        
        initGaussScalAction = QAction("Scalar", self)
        initGaussScalAction.setStatusTip("Initialize a scalar Gaussian beam.")
        initGaussScalAction.triggered.connect(self.mainWid.initSGaussBeamForm)
        makeBeamG.addAction(initGaussScalAction)

        propBeam = PhysOptMenu.addMenu("Propagate beam") 
        initPropSurfAction = QAction("To surface", self)
        initPropSurfAction.setStatusTip("Propagate a PO beam from a source surface to a target surface.")
        initPropSurfAction.triggered.connect(self.mainWid.propPOForm)
        propBeam.addAction(initPropSurfAction)
        
        initPropFFAction = QAction("To far-field", self)
        initPropSurfAction.setStatusTip("Propagate a PO beam from a source surface to a far-field surface.")
        initPropFFAction.triggered.connect(self.mainWid.propPOFFForm)
        propBeam.addAction(initPropFFAction)

        calcEffs = PhysOptMenu.addMenu("Efficiencies")
        calcSpillEffsAction = QAction("Spillover", self)
        calcSpillEffsAction.setStatusTip("Calculate spillover efficiency of a PO field.")
        calcSpillEffsAction.triggered.connect(self.mainWid.setSpillEffsForm)
        calcEffs.addAction(calcSpillEffsAction)
        
        calcTaperEffsAction = QAction("Taper", self)
        calcTaperEffsAction.setStatusTip("Calculate taper efficiency of a PO field.")
        calcTaperEffsAction.triggered.connect(self.mainWid.setTaperEffsForm)
        calcEffs.addAction(calcTaperEffsAction)
        
        calcXpolEffsAction = QAction("X-pol", self)
        calcXpolEffsAction.setStatusTip("Calculate cross-polar efficiency of a PO field.")
        calcXpolEffsAction.triggered.connect(self.mainWid.setXpolEffsForm)
        calcEffs.addAction(calcXpolEffsAction)

        calcMBEffsAction = QAction("Main beam", self)
        calcMBEffsAction.setStatusTip("Calculate main beam efficiency of a PO field.")
        calcMBEffsAction.triggered.connect(self.mainWid.setMBEffsForm)
        calcEffs.addAction(calcMBEffsAction)

        
        FocusFind = QAction("Focus finder", self)
        FocusFind.setToolTip("Calculate the focus co-ordinates of a ray-trace beam.")
        ToolsMenu.triggered.connect(self.mainWid.setFocusFindForm)
        ToolsMenu.addAction(FocusFind)
        #findRTfocusAction.triggered.connect(self.mainWid.set)

if __name__ == "__main__":

    print("lala")
    app = QApplication(sys.argv)
    win = PyPOMainWindow(parent=None)
    # def print(s):
    #     cons.appendPlainText(s)
    win.show()
    app.exec_()
