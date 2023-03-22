import threading
import time
import sys

#from PyQt5.QtCore import QObject, QThread, pyqtSignal

from src.PyPO.BindUtils import *
import src.PyPO.Config as Config
from src.PyPO.CustomLogger import CustomLogger

class Manager(object):
    def __init__(self, context, callback=None):
        self.context = context
        self.callback = callback

    def new_gthread(self, target, args, calc_type=None):
        t = threading.Thread(target=target, args=args)
        t.daemon = True
        t.start()
        
        return t
    
    def new_sthread(self, target, args):
        if self.context == "S":
            t = threading.Thread(target=target, args=args)
            t.daemon = True
            t.start()
        
            while t.is_alive(): # wait for the thread to exit
                #Config.print(f'Calculating {calc_type} {self.ws.getSymbol()}', end='\r')
                t.join(.1)


            #Config.print(f'Calculated {calc_type} in {dtime:.3f} seconds', end='\r')
            #Config.print(f'\n')
        
        else:
            target(*list(args))

    def on_thread_finished(self):
        if self.callback is not None:
            self.callback()

#class GWorker(QObject):
#    finished = pyqtSignal()
#    progress = pyqtSignal(int)
#
#    def __init__(self, target, args):
#        self.target = target
#        self.args = args
#
#    def run(self):
#        self.target(*list(self.args))
#        self.finished.emit()
#
#class GThread(QThread):
#    def __init__(self, target, args, parent=None):
#        self.parent = parent
#        self.target = target
#        self.args = args
#        self.event = threading.Event()
#       
#        super(GThread, self).__init__()
#
#    def run(self):
#        #while not self.event:
#        self.target(*list(self.args)) 
#        self.parent.on_thread_finished()
#   
#    def exit(self):
#        return
