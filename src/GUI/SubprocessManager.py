from multiprocessing import Process, Manager
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget, QMainWindow, QApplication, QVBoxLayout, QPushButton, QLabel
from Dialogs import SymDialog
from Waiter import Waiter
from time import sleep

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


    def runInSubprocess(self, work, args, propBeamDict, dialogText = ""):
        if self.subProcessRunning:
            return
        self.subProcessRunning = True

        process = Process(target = work, args = args)

        if not dialogText:
            dialogText = "Calculating"
        self.currentCalculationDialog = SymDialog(process.kill, None, dialogText) 
        self.currentCalculationDict = propBeamDict

        
        self.waiterThread = QThread(parent=self.parentWidget)
        waiter = Waiter()
        waiter.setProcess(process)
        waiter.moveToThread(self.waiterThread)
        self.waiterThread.started.connect(waiter.run)
        waiter.finished.connect(self.waiterFinished)

        process.start()
        self.waiterThread.start()
        self.currentCalculationDialog.exec_()




if __name__ == "__main__":
    import sys
    class window(QWidget):
        def __init__(self) -> None:
            super().__init__()
            layout = QVBoxLayout(self)
            btn = QPushButton("run subprocess")
            btn.clicked.connect(self.runSubprocess)
            layout.addWidget(btn)



            self.subProc = SubprocessManager(self)

        def runSubprocess(self):
            self.subProc.runInSubprocess(self.func, (5,), None)

        def func(self, x):
            for i in range(x):
                sleep(1)
                print(f"counting {i = }")

    app = QApplication(sys.argv)
    test = window()
    test.show()
    sys.exit(app.exec_())