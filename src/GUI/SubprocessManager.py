from multiprocessing import Process
from PySide2.QtCore import QThread, Signal, QObject
from src.GUI.Dialogs import SymDialog
from copy import deepcopy
import src.PyPO.System as st

##
# @file Contains the tools to run a function in a subprocess
class SubprocessManager():
    def __init__(self, parentWidget) -> None:
        self.parentWidget = parentWidget
        self.subProcessRunning = False

    ##
    # To be connected to the finished signal of the waiter object 
    # 
    # @param success Boolean, determines wether calculation finished successfully 
    def waiterFinished(self , success):
        if success:
            self.currentCalculationDialog.accept()
        else:
            self.currentCalculationDialog.reject()
        self.subProcessRunning = False

    ## 
    # Runs a function in a subprocess
    # @param work Function to be run in subprocess 
    # @param args Tuple of the arguments work takes 
    # @param dialogText String to be shown in the waiting dialog; defaults to "Calculating"
    def runInSubprocess(self, work, args, dialogText = ""):
        if self.subProcessRunning:
            return
        self.subProcessRunning = True

        process = Process(target = work, args = args)

        if not dialogText:
            dialogText = "Calculating"
        self.currentCalculationDialog = SymDialog(process.kill, None, dialogText) 

        
        self.waiterThread = QThread(parent=self.parentWidget)
        waiter = Waiter()
        waiter.setProcess(process)
        waiter.moveToThread(self.waiterThread)
        self.waiterThread.started.connect(waiter.run)
        waiter.finished.connect(self.waiterFinished)

        process.start()
        self.waiterThread.start()
        res = True if self.currentCalculationDialog.exec_() else False
        self.waiterThread.quit()
        return res
            
## 
# QObject Waits until process is finished and emits a signal
class Waiter(QObject):
    finished = Signal(int)

    def setProcess(self, process):
        self.process = process

    def run(self):
        self.process.join()
        self.finished.emit(self.process.exitcode==0)

##
# Makes a copy of system, deep copies some dictionaries.
# @param system The System object to be copied
# @param cSystem Boolean determines wether to copy the system.system dictionary
# @param cFrames list of Frames to copy
# @param cFields list of Fields to copy
# @param cCurrents list of Currents to copy
# @param cScalarFields list of ScalarFields to copy
def copySystem(system :st.System, cSystem = True, cFrames = None, cFields = None, cCurrents = None, cScalarFields = None, cAssoc = None):
    cFrames = [] if cFrames is None else cFrames
    cFields = [] if cFields is None else cFields
    cCurrents = [] if cCurrents is None else cCurrents
    cScalarFields = [] if cScalarFields is None else cScalarFields
    cAssoc = [] if cAssoc is None else cAssoc
    
    sCopy = st.System(context="G", override=False)
    if cSystem:
        sCopy.system.update(deepcopy(system.system))
    for frame in cFrames:
        sCopy.frames[frame] = system.frames[frame]
    for field in cFields:
        sCopy.fields[field] = system.fields[field]
    for current in cCurrents:
        sCopy.currents[current] = system.currents[current]
    for sField in cScalarFields:
        sCopy.scalarfields[sField] = system.scalarfields[sField]
    for sAs in cAssoc:
        sCopy.assoc[sAs] = system.assoc[sAs]
    return sCopy

