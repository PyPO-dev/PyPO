from multiprocessing import Process
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMainWindow, QApplication
from Dialogs import SymDialog
from Waiter import Waiter
from time import sleep

class SubprocessManager():
    def __init__(self) -> None:
        self.subProcessRunning = False


    def waiterFinished(self , x):
        print(f"Waiter finished {x = }")

    def runInSubprocess(self, work, args, propBeamDict):
        if self.subProcessRunning:
            return
        self.subProcessRunning = True

        process = Process(target = work, args = args)

        dialStr = f"Calculating"# {subStr} on {propBeamDict['t_name']}..."
        self.currentCalculationDialog = SymDialog(process.kill, None, dialStr) 
        self.currentCalculationDict = propBeamDict

        def calculationFinished(exitCode):
            waiter.deleteLater()
            self.waiterThread.deleteLater()
            self.waiterFinished(exitCode==0)
        
        self.waiterThread = QThread(parent=self)
        waiter = Waiter()
        waiter.setProcess(process)
        waiter.moveToThread(self.waiterThread)
        self.waiterThread.started.connect(waiter.run)
        waiter.finished.connect(calculationFinished)

        process.start()
        self.waiterThread.start()
        self.currentCalculationDialog.exec_()


def func(x):
    for i in range(x):
        sleep(1)
        print(f"working {i = }")

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    test = SubprocessManager()
    test.show()
    test.runInSubprocess(func, (5,), {1:1})
    sys.exit(app.exec_())