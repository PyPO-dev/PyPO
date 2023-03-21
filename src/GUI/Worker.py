from PyQt5.QtCore import QRunnable, QObject, pyqtSlot, pyqtSignal

class Signals(QObject):
    finished = pyqtSignal()

class Worker(QRunnable):
    def __init__(self, task, *args):
        super(Worker, self).__init__()
        self.task = task
        self.args = args
        self.signal = Signals()

        self.aborted = False

    @pyqtSlot()
    def run(self):
        self.task(*list(self.args))
        self.signal.finished.emit()

    #@pyqtSlot()
    def abort(self):
        self.aborted = True

    def stop(self):
        self._stop_event.set()
    
    def kill(self):
        self.is_killed = True

class GWorker(QObject):
    finished = pyqtSignal()

    def __init__(self, task, *args):
        super(GWorker, self).__init__()
        self.task = task
        self.args = args

    #@pyqtSlot()
    def run(self):
        self.task(*list(self.args))
        self.finished.emit()
    
    def kill(self):
        self.is_killed = True
