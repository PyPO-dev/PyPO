import os
import sys
import shutil
from time import time
from threading import Thread, Event
from traceback import print_tb
from multiprocessing import Process, Manager

from PyQt5.QtWidgets import QApplication, QLabel, QTextEdit, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt, QThreadPool, QThread

from src.GUI.ParameterForms import formGenerator
from src.GUI.ParameterForms.InputDescription import InputDescription
from src.GUI.utils import inType
import src.GUI.ParameterForms.formData as fData
from src.GUI.PlotScreen import PlotScreen
from src.GUI.Accordion import Accordion
from src.GUI.Dialogs import SymDialog
from src.GUI.Worker import Worker, GWorker
from src.GUI.Waiter import Waiter
from src.GUI.WorkSpace import Workspace

from src.PyPO.CustomLogger import CustomGUILogger

# import numpy as np
from src.PyPO.Checks import InputReflError, InputRTError ##TODO @arend is this useful?

sys.path.append('../')
sys.path.append('../../')
import src.PyPO.System as st
import src.PyPO.Threadmgr as TManager
import src.PyPO.Checks as chk

##
# @file 
# defines classes PyPOMainWindow and MainWidget
# PyPOMainWindow is responsible for setting up the window and toolbars
# MainWidget is responsible for all gui functionalities
#
class MainWidget(QWidget):
    ##
    # Constructor. Configures the layout and initializes the underlying system
    # @see System
    # 
    # 

    def __init__(self, parent=None):
        super().__init__(parent)
        # Window settings
        self.setWindowTitle("PyPO")



        # GridParameters
        self.grid = QGridLayout()
        self.GPWorkSpace      = [0, 0, 2, 1]
        self.GPParameterForm  = [0, 1, 2, 1]
        self.GPPlotScreen     = [0, 2, 1, 1]
        self.GPConsole        = [1, 2, 1, 1]

        self._mkConsole()
        # init System
        self.clog_mgr = CustomGUILogger(os.path.basename(__file__))
        self.clog = self.clog_mgr.getCustomGUILogger(self.console)
        #self.clog.info(f"STARTED PyPO GUI SESSION.")
        
        self.stm = st.System(redirect=self.clog, context="G")
       
        self.clog = self.stm.getSystemLogger()
        self.clog.info(f"STARTED PyPO GUI SESSION.")

        # init layout
        self.grid.setContentsMargins(5,5,5,5)
        # self.grid.setMargin(0)
        self.grid.setSpacing(5)

        self._mkWorkSpace()
        self._mkPlotScreen()
        self.setLayout(self.grid)
        
        self.event_stop = Event()

        self.threadpool = QThreadPool()

        # NOTE Raytrace stuff
        self.frameDict = {}
        # end NOTE

    ### Gui setup functions
    ##
    # @guiSetup
    # Adds a widget to the layout of PyPOMainWidget
    def addToWindowGrid(self, widget, param, vStretch= 0, hStretch= 0):
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid.addWidget(widget, param[0], param[1], param[2], param[3])

        if vStretch:
            self.grid.setRowStretch(param[0], vStretch)
        if hStretch:
            self.grid.setColumnStretch(param[1],hStretch)


    ##  
    # @guiSetup
    # Configures the console widget
    # 
    def _mkConsole(self):
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.addToWindowGrid(self.console, self.GPConsole, vStretch=1, hStretch=3)
        self.cursor = QTextCursor(self.console.document())
        
        
        # self.console.appendPlainText("********** PyPO Console **********")
        # self.addToWindowGrid(self.console, self.GPConsole)
    
    ##
    # @guiSetup
    # constructs the elements column on the left side of the screen
    # 
    def _mkWorkSpace(self):
        # delete if exists
        if hasattr(self, "WorkSpace"):
            self.WorkSpace.setParent(None)
        # rebuild 
        self.WorkSpace = Workspace()
        self.addToWindowGrid(self.WorkSpace, self.GPWorkSpace, hStretch=1)

    ##
    # @guiSetup
    # Constructs the tab widget which will later hold plots
    def _mkPlotScreen(self):
        self.PlotWidget = QTabWidget()
        self.PlotWidget.setTabsClosable(True)
        self.PlotWidget.setTabShape(QTabWidget.Rounded)
        self.PlotWidget.tabCloseRequested.connect(self.closePlotTab)
        # self.PlotWidget.setMaximumHeight(700)
        self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen,vStretch=2, hStretch=3)

    ##
    # @guiSetup
    # Generates a form widget
    # 
    # @param formData List of InputDescription objects
    # @param readAction Function to be called when forms ok-button is clicked
    # 
    def setForm(self, formData, readAction, okText=None):
        if hasattr(self, "ParameterWid"):
            try:
                self.ParameterWid.setParent(None)
            except:
                pass
        self.ParameterWid = formGenerator.FormGenerator(formData, readAction, okText=okText)

        self.ParameterWid.closed.connect(self.removeForm)

        self.formScroll = QScrollArea()
        self.formScroll.setWidget(self.ParameterWid)
        self.formScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.formScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # scroll.border
        # self.formScroll.setWidgetResizable(True)
        self.formScroll.setContentsMargins(0,0,0,0)

        self.addToWindowGrid(self.formScroll,self.GPParameterForm)
    
    ## 
    # removes the form ParameterWid if exists 
    def removeForm(self):
        if hasattr(self, "formScroll"):
            print("removingForm")
            self.formScroll.setParent(None)
            self.formScroll.deleteLater()
            self.ParameterWid.setParent(None)
            self.ParameterWid.deleteLater()
    
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
    # plots a group from the System
    # 
    # @param group str Name of the group in system
    # 
    def plotGroup(self, group):
        if self.stm.groups:
            figure, _ = self.stm.plotGroup(group, show=False, ret=True)
        else :
            figure = None
        self.addPlot(figure, group)

    ##
    # Generate a snapshot form.
    def snapActionForm(self, element):
        self.setForm(fData.snapForm(element, list(self.stm.system[element]["snapshots"].keys()), "element"), readAction=self.snapAction, okText="Take snapshot")
    
    ##
    # Generate a snapshot form for a group.
    def snapGroupActionForm(self, group):
        self.setForm(fData.snapForm(group, list(self.stm.groups[group]["snapshots"].keys()), "group"), readAction=self.snapAction, okText="Take snapshot")
 
    ##
    # Generate a snapshot form for ray-trace frame.
    def snapFrameActionForm(self, frame):
        self.setForm(fData.snapForm(frame, list(self.stm.frames.snapshots.keys()), "frame"), readAction=self.snapAction, okText="Take snapshot")
    
    ##
    # Take, revert or delete a snapshot
    # 
    # @param element Name of the surface in system
    # 
    def snapAction(self):
        try:
            snapDict = self.ParameterWid.read()
            if snapDict["options"] == "Take":
                self.stm.snapObj(snapDict["name"], snapDict["snap_name"], snapDict["obj"])
            
            elif snapDict["options"] == "Revert":
                self.stm.revertToSnap(snapDict["name"], snapDict["snap_name"], snapDict["obj"])
                
            elif snapDict["options"] == "Delete":
                self.stm.deleteSnap(snapDict["name"], snapDict["snap_name"], snapDict["obj"])

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


    ##
    # Generate a copy form for a element.
    #
    # @param element Name of element to be copied.
    def copyElementActionForm(self, element):
        self.setForm(fData.copyForm(element), readAction=self.copyElementAction, okText="Make Copy")

    ##
    # Copy a element in system to a new version.
    def copyElementAction(self):
        try:
            copyDict = self.ParameterWid.read()
            
            self.stm.copyElement(copyDict["name"], copyDict["name_copy"])
            self.addReflectorWidget(copyDict["name_copy"])


        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    ##
    # Generate a copy form for a group.
    #
    # @param group Name of group to be copied.
    def copyGroupActionForm(self, group):
        self.setForm(fData.copyForm(group), readAction=self.copyGroupAction, okText="Make Copy")

    ##
    # Copy a group in system to a new version.
    def copyGroupAction(self):
        try:
            copyDict = self.ParameterWid.read()
            
            self.stm.copyGroup(copyDict["name"], copyDict["name_copy"])
            self.addGroupWidget(copyDict["name_copy"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
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
        frameList = []

        if self.stm.frames:
            for key in self.stm.frames.keys():
                frameList.append(key)
        
        if self.stm.system:
            figure, _ = self.stm.plotSystem(ret = True, show=False, save=False, RTframes=frameList)
        
        else:
            figure = None
        self.addPlot(figure,"Ray Trace Frame %d" %(self.getRayPlotNr()))

    ### Functionalities: Systems 


    ##
    # opens a form that allows user to save the System
    def saveSystemForm(self):
        self.setForm(fData.saveSystemForm(), readAction=self.saveSystemAction, okText="Save System")
    
    ##
    # Saves the current system state under the name given in form
    def saveSystemAction(self):
        try:
            saveDict = self.ParameterWid.read()
            self.stm.saveSystem(saveDict["name"]) 
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    ##
    # opens a form that allows user to delete a saved System
    def deleteSavedSystemForm(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fData.loadSystemForm(systemList), readAction=self.deleteSavedSystemAction, okText="Delete System")

    ##
    # Deletes system selected in form
    def deleteSavedSystemAction(self):
        try:
            removeDict = self.ParameterWid.read()
            shutil.rmtree(os.path.join(self.stm.savePathSystems, removeDict["name"]))
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    ##
    # opens a form that allows user to load a saved System
    def loadSystemForm(self):
        systemList = [os.path.split(x[0])[-1] for x in os.walk(self.stm.savePathSystems) if os.path.split(x[0])[-1] != "systems"]
        self.setForm(fData.loadSystemForm(systemList), readAction=self.loadSystemAction, okText="Load System")
    
    ##
    # Loads system selected in from form
    def loadSystemAction(self):
        try:
            loadDict = self.ParameterWid.read()
            self._mkWorkSpace()

            self.stm.loadSystem(loadDict["name"]) 
            self.refreshWorkspaceSection(self.stm.system, "elements")
            self.refreshWorkspaceSection(self.stm.frames, "frames")
            self.refreshWorkspaceSection(self.stm.fields, "fields")
            self.refreshWorkspaceSection(self.stm.currents, "currents")
            self.refreshWorkspaceSection(self.stm.scalarfields, "scalarfields")
            self.refreshWorkspaceSection(self.stm.groups, "groups")

            self.removeForm()
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # removes an element from the system
    # @param element Name of the element in the system
    def removeElement(self, element):
        try:
            self.stm.removeElement(element)
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    def removeFrame(self, frame):
        try:
            self.stm.removeFrame(frame)
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    def addGroupForm(self):
        elements = []
        # print(self.stm.system)
        for element in self.stm.system.values():
            # print(element)
            if element["gmode"] != 2:
                elements.append(element["name"])
        self.setForm(fData.addGroupForm(elements), self.addGroupAction)

    def addGroupAction(self):
        try:
            groupDict = self.ParameterWid.read()
            self.stm.groupElements(groupDict["name"], *groupDict["selected"])

            self.addGroupWidget(groupDict["name"])


        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err) 






    ##
    # TODO: @Maikel Rename this function and evaluate its necessity
    def refreshWorkspaceSection(self, columnDict, section):
        for key in columnDict.keys():
            if section == "elements":
                self.addReflectorWidget(key)
            
            elif section == "frames":
                self.addFrameWidget(key)
            
            elif section == "fields":
                self.addFieldWidget(key)
            
            elif section == "currents":
                self.addCurrentWidget(key)

            elif section == "scalarfields":
                self.addSFieldWidget(key)
                
            elif section == "groups":
                self.addGroupWidget(key)

    ### Functionalities: Adding widgets to workspace
    # TODO:doc
    def addReflectorWidget(self, name):
        self.WorkSpace.addReflector(name, self.removeElement, 
                                         self.transformSingleForm, self.plotElement, 
                                         self.snapActionForm, self.copyElementActionForm)
    
    def addGroupWidget(self, name):
        self.WorkSpace.addGroup(name, self.stm.removeGroup, self.plotGroup, self.transformGroupForm, self.snapGroupActionForm, self.copyGroupActionForm)
    
    def addFrameWidget(self, name):
        self.WorkSpace.addRayTraceFrames(name, self.removeFrame, 
                                              self.transformFrameForm, self.plotFrameForm,  
                                              self.calcRMSfromFrame, self.snapFrameActionForm)
        

    def addFieldWidget(self, name):
        self.WorkSpace.addFields(name, self.stm.removeField, self.plotFieldForm)

    def addCurrentWidget(self, name):
        self.WorkSpace.addCurrent(name, self.stm.removeCurrent, self.plotCurrentForm)

    def addSFieldWidget(self, name):
        self.WorkSpace.addSPOFields(name, self.stm.removeScalarField, self.plotSFieldForm)
    
    ### Functionalities: Adding Elements 

    ##
    # Shows form to add a plane
    def addPlaneForm(self):
        self.setForm(fData.makePlaneInp(), readAction=self.addPlaneAction)

    ##
    # Reads form and adds plane to System
    def addPlaneAction(self):
        try:
            elementDict = self.ParameterWid.read()

            self.stm.addPlane(elementDict) 
            self.addReflectorWidget(elementDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    ##
    # Shows from to add a quadric surface
    def addQuadricForm(self):
        self.setForm(fData.makeQuadricSurfaceInp(), readAction=self.addQuadricAction)
    
    ##
    # Reads quadric form, evaluates surface type and calls corresponding add____Action
    def addQuadricAction(self):
        try:
            elementDict = self.ParameterWid.read()
            if elementDict["type"] == "Parabola":
                self.stm.addParabola(elementDict)
                # print(f"PARABOLA{ elementDict['name'] }")
            elif elementDict["type"] == "Hyperbola":
                self.stm.addHyperbola(elementDict)
            elif elementDict["type"] == "Ellipse":
                self.stm.addEllipse(elementDict)
            self.addReflectorWidget(elementDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Reads form and adds parabola to System
    def addParabolaAction(self):
        try:
            elementDict = self.ParameterWid.read()

            self.stm.addParabola(elementDict) 
            self.addReflectorWidget(elementDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Reads form and adds hyperbola to System
    def addHyperbolaAction(self):
        try:
            elementDict = self.ParameterWid.read()
        
            self.stm.addHyperbola(elementDict) 
            self.addReflectorWidget(elementDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Reads form and adds ellipse to System
    def addEllipseAction(self):
        try:
            elementDict = self.ParameterWid.read()
        
            self.stm.addEllipse(elementDict) 
            self.addReflectorWidget(elementDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ### Functionalities: Transforming Elements 

    ##
    # Shows single element transformation form
    def transformSingleForm(self, element):
        self.setForm(fData.makeTransformationForm(element), self.transformAction, okText="Apply transformation")
    
    ##
    # Shows single element transformation form
    def transformFrameForm(self, frame):
        self.setForm(fData.makeTransformationForm(frame, obj="frame"), self.transformFrameAction, okText="Apply transformation")
    
    ##
    # Applies single element transformation
    def transformAction(self, element):
        try:
            dd = self.ParameterWid.read()
            transformationType = dd["type"]
            vector = dd["vector"]

            if transformationType == "Translation":
                self.stm.translateGrids(dd["element"], vector, mode=dd["mode"].lower())
                # print(f'Translated {dd["element"]} by {self._formatVector(vector)} mm')
            elif transformationType == "Rotation":
                self.stm.rotateGrids(dd["element"], vector, pivot=dd["pivot"], mode=dd["mode"].lower())
                # print(f'Rotated {dd["element"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["pivot"])}')
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ### Functionalities: Transforming Groups 

    ##
    # Shows group transformation form
    def transformGroupForm(self, group):
        self.setForm(fData.makeTransformationForm(group, obj="group"), self.transformGroupAction, okText="Apply transformation")
    
    ##
    # Applies group transformation
    def transformGroupAction(self, element):
        try:
            dd = self.ParameterWid.read()
            transformationType = dd["type"]
            vector = dd["vector"]

            if transformationType == "Translation":
                self.stm.translateGrids(dd["group"], vector, mode=dd["mode"].lower(), obj="group")
            elif transformationType == "Rotation":
                self.stm.rotateGrids(dd["group"], vector, pivot=dd["pivot"], mode=dd["mode"].lower(), obj="group")
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    ##
    # Applies single frame transformation
    def transformFrameAction(self, frame):
        try:
            dd = self.ParameterWid.read()
            transformationType = dd["type"]
            vector = dd["vector"]
        
            if transformationType == "Translation":
                self.stm.translateGrids(dd["frame"], vector, mode=dd["mode"].lower(), obj="frame")
                # print(f'Translated {dd["frame"]} by {self._formatVector(vector)} mm')
            elif transformationType == "Rotation":
                self.stm.rotateGrids(dd["frame"], vector, pivot=dd["pivot"], mode=dd["mode"].lower(), obj="frame")
                # print(f'Rotated {dd["frame"]} by {self._formatVector(vector)} deg around {self._formatVector(dd["pivot"])}')
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    ##
    # Shows multiple element transformation form
    def transformationMultipleForm(self):
        movableElements = []
        for key, elem in self.stm.system.items():
            if elem["gmode"] != 2:
                movableElements.append(key)

        self.setForm(
            [InputDescription(inType.elementSelector, "elements", options=movableElements)]+
            fData.makeTransformationElementsForm(self.stm.system.keys()), self.transformationMultipleAction
            , okText="Apply transformation")

    ##
    # Applies multiple element transformation
    def transformationMultipleAction(self):
        try:
            transfDict = self.ParameterWid.read()

            if transfDict["type"] == "Translation":
                self.stm.translateGrids(transfDict["elements"], transfDict["vector"])

            if transfDict["type"] == "Rotation":
                self.stm.rotateGrids(transfDict["elements"], transfDict["vector"], transfDict["pivot"])

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    ### Functionalities: TRFrames 
    #NOTE Raytrace widgets

    ##
    # Shows tube frame form
    def initTubeFrameForm(self):
        self.setForm(fData.initTubeFrameInp(), readAction=self.initTubeFrameAction, okText="Add frame")
    
    ##
    # Reads form and adds a tube frame to system
    def initTubeFrameAction(self):
        try:
            RTDict = self.ParameterWid.read()
        
            self.stm.createTubeFrame(RTDict)
            self.addFrameWidget(RTDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Shows form to initialize gaussian frame 
    def initGaussianFrameForm(self):
        self.setForm(fData.initGaussianFrameInp(), readAction=self.initGaussianFrameAction, okText="Add frame")
    

    ##
    # Reads form and adds a gaussian frame to system
    def initGaussianFrameAction(self):
        try:
            GRTDict = self.ParameterWid.read()

            if not "seed" in GRTDict.keys():
                GRTDict["seed"] = -1

      
            dialStr = f"Calculating Gaussian ray-trace frame {GRTDict['name']}..."
            worker = Worker(self.stm.createGRTFrame, GRTDict)
            dial = SymDialog(worker.kill, self.clog, dialStr) 

            worker.signal.finished.connect(dial.accept) 
            worker.signal.finished.connect(lambda : self.addFrameWidget(GRTDict["name"])) 

            self.threadpool.start(worker)
            dial.exec_()
        
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Shows form to propagate rays
    def setPropRaysForm(self):
        self.setForm(fData.propRaysInp(self.stm.frames, self.stm.system), self.addPropRaysAction, okText="Propagate rays")

    ##
    # Reads form and propagates rays
    def addPropRaysAction(self): 
        try:
            propRaysDict = self.ParameterWid.read()
        
            self.stm.runRayTracer(propRaysDict)
            self.addFrameWidget(propRaysDict["fr_out"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Shows form to plot preselected a frame
    # 
    # @param frame Frame to plot
    def plotFrameForm(self, frame):
        self.setForm(fData.plotFrameOpt(frame), readAction=self.addPlotFrameAction, okText="Plot frame")

    ##
    # Reads form and plots frame
    def addPlotFrameAction(self):
        try:
            plotFrameDict = self.ParameterWid.read()
            fig = self.stm.plotRTframe(plotFrameDict["frame"], project=plotFrameDict["project"], ret=True)
            self.addPlot(fig, f'{plotFrameDict["frame"]} - {plotFrameDict["project"]}')

            self.addToWindowGrid(self.PlotWidget, self.GPPlotScreen)

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


    ### Functionalities: PO 
    ##
    # Shows form to initialize gaussian beam 
    def initGaussBeamForm(self):
        self.setForm(fData.initGaussianInp(self.stm.system), readAction=self.initGaussBeamAction, okText="Add beam")

    ##
    # Reads form and adds a vectorial gaussian beam to system
    def initGaussBeamAction(self):
        try:
            GDict = self.ParameterWid.read()
        
            self.stm.createGaussian(GDict, GDict["surface"])
            self.addFieldWidget(GDict["name"])
            self.addCurrentWidget(GDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    ##
    # Shows form to initialize a scalar gaussian beam TODO: is this correct??
    def initSGaussBeamForm(self):
        self.setForm(fData.initSGaussianInp(self.stm.system), readAction=self.initSGaussBeamAction, okText="Add beam")
    
    ##
    # Reads form and adds a scalar gaussian beam to system
    def initSGaussBeamAction(self):
        try:
            GDict = self.ParameterWid.read()
            
            self.stm.createScalarGaussian(GDict, GDict["surface"])
            self.addSFieldWidget(GDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    ##
    # Shows form to initialize a physical optics propagation
    def initPSBeamForm(self):
        self.setForm(fData.initPSInp(self.stm.system), readAction=self.initPSBeamAction, okText="Add beam")
    
    
    ##
    # Reads form and adds a vectorial point source beam to system
    def initPSBeamAction(self):
        try:
            PSDict = self.ParameterWid.read()
            
            self.stm.generatePointSource(PSDict, PSDict["surface"])
            self.addFieldWidget(PSDict["name"])
            self.addCurrentWidget(PSDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Shows form to initialize a scalar point source beam
    def initSPSBeamForm(self):
        self.setForm(fData.initSPSInp(self.stm.system), readAction=self.initSPSBeamAction, okText="Add beam")
    

    ##
    # Reads form and adds a scalar point source beam to system
    def initSPSBeamAction(self):
        try:
            SPSDict = self.ParameterWid.read()
            
            self.stm.generatePointSourceScalar(SPSDict, SPSDict["surface"])
            self.addSFieldWidget(SPSDict["name"])
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # Shows form to plot field
    #
    # @param field Field to plot
    def plotFieldForm(self, field):
        if self.stm.system[self.stm.fields[field].surf]["gmode"] == 2:
            self.setForm(fData.plotFarField(field), readAction=self.plotFieldAction, okText="Plot")
        else:
            self.setForm(fData.plotField(field), readAction=self.plotFieldAction, okText="Plot")

    ##
    # Reads form and plots field
    def plotFieldAction(self):
        try:
            plotFieldDict = self.ParameterWid.read()
            fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], plotFieldDict["comp"], 
                                        project=plotFieldDict["project"], ret=True)
            self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    ##
    # TODO: whats the difference with above? Perhapses we should rename function
    #
    # @param field Field to plot
    def plotSFieldForm(self, field):
        self.setForm(fData.plotSField(field, self.stm.system[self.stm.scalarfields[field].surf]["gmode"]), readAction=self.plotSFieldAction, okText="Plot")

    ##
    # Reads form and plots scalar field
    def plotSFieldAction(self):
        try:
            plotSFieldDict = self.ParameterWid.read()
            fig, _ = self.stm.plotBeam2D(plotSFieldDict["field"], 
                                        project=plotSFieldDict["project"], ret=True)
            self.addPlot(fig, f'{plotSFieldDict["field"]} - {plotSFieldDict["project"]}')

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    
    ## 
    # Shows form to plot current
    #
    # @param current Current to plot
    def plotCurrentForm(self, current):
        self.setForm(fData.plotCurrentOpt(current), readAction=self.plotCurrentAction, okText="Plot")
    
    ##
    # Reads form and plots current
    def plotCurrentAction(self):
        try:
            plotFieldDict = self.ParameterWid.read()
            fig, _ = self.stm.plotBeam2D(plotFieldDict["field"], 
                                        plotFieldDict["comp"], project=plotFieldDict["project"], ret=True)
            self.addPlot(fig, f'{plotFieldDict["field"]} - {plotFieldDict["comp"]}  - {plotFieldDict["project"]}')

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    
    ##
    # Shows form to propagate physical optics beam to surface 
    def propPOForm(self):
        self.setForm(fData.propPOInp(self.stm.currents, self.stm.scalarfields, self.stm.system), self.propPOAction, okText="Propagate beam")
    
    ##
    # Shows form to propagate physical optics beam far field 
    def propPOFFForm(self):
        self.setForm(fData.propPOFFInp(self.stm.currents, self.stm.system), self.propPOAction, okText="Propagate beam")
   
    def _addToWidgets(self, propBeamDict):
        print(f"{propBeamDict = }")
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

    # def _addToWidgetsOfCalc(self):
    #     self._addToWidgets()


    def runPOWorker(self, s_copy, runPODict, returnDict):
        try:
            s_copy.runPO(runPODict)

            returnDict["system"] = s_copy
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)

    def waiterFinished(self):
        # process.join()
        try:
            print("waiter: Process joined")

            self.currentCalculationDialog.accept()
            self._addToWidgets(self.currentCalculationDict)
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    ##
    # Reads form propagates beam, runs calculation on another thread
    def propPOAction(self):
        try:
            propBeamDict = self.ParameterWid.read()
            print(propBeamDict)
            
            chk.check_runPODict(propBeamDict, self.stm.system.keys(), self.stm.fields.keys(), self.stm.currents.keys(),
                            self.stm.scalarfields.keys(), self.stm.frames.keys(), self.clog)
        
            if propBeamDict["mode"] == "scalar":
                subStr = "scalar field"
            else:
                subStr = propBeamDict["mode"]

            start_time = time()
        
            self.clog.info("*** Starting PO propagation ***")


            s_copy = st.System(context="G")
            
            s_copy.system[propBeamDict["t_name"]] = self.stm.system[propBeamDict["t_name"]]
            if "s_current" in propBeamDict:
                s_copy.currents[propBeamDict["s_current"]] = self.stm.currents[propBeamDict["s_current"]]
                s_copy.system[self.stm.currents[propBeamDict["s_current"]].surf] = self.stm.system[self.stm.currents[propBeamDict["s_current"]].surf]
                print(f"{self.stm.system[self.stm.currents[propBeamDict['s_current']].surf] = }")
            elif "s_field" in propBeamDict:
                s_copy.scalarfields[propBeamDict["s_field"]] = self.stm.scalarfields[propBeamDict["s_scalarfield"]]
                s_copy.system[self.stm.scalarfields[propBeamDict["s_field"]].surf] = self.stm.system[self.stm.scalarfields[propBeamDict["s_field"]].surf]

            mgr = Manager()
            returnDict = mgr.dict()

            args = (s_copy, propBeamDict, returnDict)
            process = Process(target = self.runPOWorker, args = args)

            dialStr = f"Calculating {subStr} on {propBeamDict['t_name']}..."
            self.currentCalculationDialog = SymDialog(process.kill, self.clog, dialStr) 
            self.currentCalculationDict = propBeamDict

            waiterTread = QThread(parent=self)
            waiter = Waiter()
            waiter.setProcess(process)
            waiter.moveToThread(waiterTread)
            waiterTread.started.connect(waiter.run)
            waiter.finished.connect(waiter.deleteLater)
            # waiter.finished.connect(waiterTread.quit)
            # waiter.finished.connect(waiterTread.deleteLater)
            waiter.finished.connect(self.waiterFinished)

            process.start()
            waiterTread.start()
            if self.currentCalculationDialog.exec_():
                s_copy = returnDict["system"]
                print(f"{s_copy = }")
                self.stm.frames.update(s_copy.frames)
                self.stm.fields.update(s_copy.fields)
                self.stm.currents.update(s_copy.currents)
                self.stm.scalarfields.update(s_copy.scalarfields)

                dtime = time() - start_time
                self.clog.info(f"*** Finished: {dtime:.3f} seconds ***")

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(f"PO Propagation did not end successfully: {err}.")
        
    
    #
    ##TODO Unite efficiencies
    ##
    # Shows form to calculate taper efficiencies
    def setTaperEffsForm(self):
        self.setForm(fData.calcTaperEff(self.stm.fields, self.stm.system), self.calcTaperAction, okText="Calculate")
    
    ##
    # Shows form to calculate spillover efficiencies
    def setSpillEffsForm(self):
        self.setForm(fData.calcSpillEff(self.stm.fields, self.stm.system), self.calcSpillAction, okText="Calculate")

    ##
    # Shows form to calculate x-pol efficiencies TODO: x-pol
    def setXpolEffsForm(self):
        self.setForm(fData.calcXpolEff(self.stm.fields, self.stm.system), self.calcXpolAction, okText="Calculate")

    ##
    # Shows form to calculate main beam efficiencies
    def setMBEffsForm(self):
        self.setForm(fData.calcMBEff(self.stm.fields, self.stm.system), self.calcMBAction, okText="Calculate")
    
    ##
    # Reads form and calculates taper efficiencies
    def calcTaperAction(self):
        try:
            TaperDict = self.ParameterWid.read()
            eff_taper = self.stm.calcTaper(TaperDict["f_name"], TaperDict["comp"])
            self.clog.info(f'Taper efficiency of {TaperDict["f_name"]}, component {TaperDict["comp"]} : {eff_taper}')
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


    ##
    # Reads form and calculates spillover efficiencies
    def calcSpillAction(self):
        try:
            SpillDict = self.ParameterWid.read()

            aperDict = {
                    "center"    : SpillDict["center"],
                    "inner"      : SpillDict["inner"],
                    "outer"      : SpillDict["outer"]
                    }

            eff_spill = self.stm.calcSpillover(SpillDict["f_name"], SpillDict["comp"], aperDict)
            self.clog.info(f'Spillover efficiency of {SpillDict["f_name"]}, component {SpillDict["comp"]} : {eff_spill}')
        
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


    ##
    # Reads form and calculates x-pol efficiencies TODO: x-pol?
    def calcXpolAction(self):
        try:
            XpolDict = self.ParameterWid.read()
            eff_Xpol = self.stm.calcXpol(XpolDict["f_name"], XpolDict["co_comp"], XpolDict["cr_comp"])
            self.clog.info(f'X-pol efficiency of {XpolDict["f_name"]}, co-component {XpolDict["co_comp"]} and X-component {XpolDict["cr_comp"]} : {eff_Xpol}')
        
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


    ##
    # Reads form and calculates main beam efficiencies
    def calcMBAction(self):
        try:
            MBDict = self.ParameterWid.read()
            eff_mb = self.stm.calcMainBeam(MBDict["f_name"], MBDict["comp"], MBDict["thres"], MBDict["mode"])
            self.clog.info(f'Main beam efficiency of {MBDict["f_name"]}, component {MBDict["comp"]} : {eff_mb}')
            self.addSFieldWidget(f"fitGauss_{MBDict['f_name']}")
        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)
    
    #END NOTE
    
    ##
    # calculates root mean square of a frame
    def calcRMSfromFrame(self, frame):
        rms = self.stm.calcSpotRMS(frame)
        self.clog.info(f"RMS value of {frame} : {rms} mm")

    def setFocusFindForm(self):
        self.setForm(fData.focusFind(list(self.stm.frames.keys())), self.findFocusAction, okText="Find focus")

    def findFocusAction(self):
        try:
            findFocusDict = self.ParameterWid.read()
            focus = self.stm.findRTfocus(findFocusDict["name_frame"], verbose=True)
            self.addReflectorWidget(f"focal_plane_{findFocusDict['name_frame']}")
            self.addFrameWidget(f"focus_{findFocusDict['name_frame']}")
            
            self.clog.info(f"Focus of {findFocusDict['name_frame']} : {focus}")

        except Exception as err:
            print(err)
            print_tb(err.__traceback__)
            self.clog.error(err)


class PyPOMainWindow(QMainWindow):
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("PyPO")
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

        # transformElementsAction = QAction("Transform elements", self) ##TODO deprecated? There is no longer a form for this action @arend
        # transformElementsAction.setStatusTip("Transform a group of elements.")
        # transformElementsAction.triggered.connect(self.mainWid.transformationMultipleForm)
        # ElementsMenu.addAction(transformElementsAction)

        # ElementsMenu.addSeparator()
        
        addGroupAction = QAction("Add group", self)
        addGroupAction.setStatusTip("Adds showsForm to add group")
        addGroupAction.triggered.connect(self.mainWid.addGroupForm)
        ElementsMenu.addAction(addGroupAction)
        
    ### System actions
        # newSystem = QAction('Add System', self)
        # newSystem.triggered.connect(self.mainWid.addSystemAction)
        # SystemsMenu.addAction(newSystem)

        plotSystem = QAction("Plot system", self)
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

        calcEffs = ToolsMenu.addMenu("Efficiencies")
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
        FocusFind.triggered.connect(self.mainWid.setFocusFindForm)
        ToolsMenu.addAction(FocusFind)


        #findRTfocusAction.triggered.connect(self.mainWid.set)


