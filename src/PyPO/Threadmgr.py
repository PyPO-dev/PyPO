import threading
import time
import sys

#from src.PyPO.BindUtils import *
#import src.PyPO.Config as Config

class Manager(object):
    def __init__(self, context, callback=None):
        self.context = context
        self.callback = callback
    
    def new_sthread(self, target, args):
        if self.context == "S":
            t = threading.Thread(target=target, args=args)
            t.daemon = True
            t.start()
        
            while t.is_alive(): # wait for the thread to exit
                t.join(.1)

        else:
            target(*list(args))

    def on_thread_finished(self):
        if self.callback is not None:
            self.callback()
