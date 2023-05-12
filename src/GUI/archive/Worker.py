from PySide2.QtCore import QRunnable, QObject, Slot, Signal

class Signals(QObject):
    finished = Signal()

class Worker(QRunnable):
    def __init__(self, task, *args):
        super(Worker, self).__init__()
        self.task = task
        self.args = args
        self.signal = Signals()

        self.aborted = False

    @Slot()
    def run(self):
        self.task(*list(self.args))
        self.signal.finished.emit()

    #@Slot()
    def abort(self):
        self.aborted = True

    def stop(self):
        self._stop_event.set()
    
    def kill(self):
        self.is_killed = True

class GWorker(QObject):
    finished = Signal()

    def __init__(self, task, *args):
        super(GWorker, self).__init__()
        self.task = task
        self.args = args

    #@Slot()
    def run(self):
        self.task(*list(self.args))
        self.finished.emit()
    
    def kill(self):
        self.is_killed = True
