import threading
import time
import sys

from src.PyPO.BindUtils import *
import src.PyPO.Config as Config

class Manager(object):
    def __init__(self, context, callback=None):
        self.context = context
        self.ws = WaitSymbol()
        self.callback = callback

    def new_gthread(self, target, args, calc_type):
        start_time = time.time()
        t = GThread(target=target, args=args, parent=self)
        t.daemon = True
        t.start()
        """ 
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating {calc_type} {self.ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated {calc_type} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')
        """

        return t
    
    def new_sthread(self, target, args, calc_type):
        if self.context == "S":
            start_time = time.time()
            t = threading.Thread(target=target, args=args)
            t.daemon = True
            t.start()
        
            while t.is_alive(): # wait for the thread to exit
                #Config.print(np.array([1,0,0]))
                Config.print(f'Calculating {calc_type} {self.ws.getSymbol()}', end='\r')
                t.join(.1)


            dtime = time.time() - start_time
            Config.print(f'Calculated {calc_type} in {dtime:.3f} seconds', end='\r')
            Config.print(f'\n')
        
        else:
            target(*list(args))

    def on_thread_finished(self):
        if self.callback is not None:
            self.callback()

class GThread(threading.Thread):
    def __init__(self, target, args, parent=None):
        self.parent = parent
        self.target = target
        self.args = args
        self.event = threading.Event()
       
        super(GThread, self).__init__()

    def run(self):
        #while not self.event:
        self.target(*list(self.args)) 
        self.parent.on_thread_finished()
   
    def exit(self):
        return
