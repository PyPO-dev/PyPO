from multiprocessing import Process, Manager
from PySide2.QtCore import QThread, Signal, QObject
from src.GUI.Dialogs import SymDialog
from time import sleep
from copy import deepcopy
import src.PyPO.System as st


class SubprocessManager():
    def __init__(self, parentWidget) -> None:
        self.parentWidget = parentWidget
        self.subProcessRunning = False


    def waiterFinished(self , success):
        if success:
            self.currentCalculationDialog.accept()
        else:
            self.currentCalculationDialog.reject()
        self.subProcessRunning = False


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
            
class Waiter(QObject):
    finished = Signal(int)

    def setProcess(self, process):
        self.process = process

    def run(self):
        self.process.join()
        print("waiter: Process joined")
        self.finished.emit(self.process.exitcode==0)


def copySystem(system :st.System, cSystem = True, cFrames = None, cFields = None, cCurrents = None, cScalarFields = None):
    cFrames = [] if cFrames is None else cFrames
    cFields = [] if cFields is None else cFields
    cCurrents = [] if cCurrents is None else cCurrents
    cScalarFields = [] if cScalarFields is None else cScalarFields
    
    s2 = st.System(context="G", override=False)
    if cSystem:
        s2.system.update(deepcopy(system.system))
    for frame in cFrames:
        s2.frames[frame] = system.frames[frame]
    for field in cFields:
        s2.fields[field] = system.fields[field]
    for current in cCurrents:
        s2.currents[current] = system.currents[current]
    for sField in cScalarFields:
        s2.scalarfields[sField] = system.scalarfields[sField]
    return s2

