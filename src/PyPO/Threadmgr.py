import threading

##
# @file 
# File containing the threadmanager class for PyPO.
# This class is responsible for launching heavy calculations on a separate daemon thread,
# preventing the program from becoming unresponsive.

class Manager(object):
    """!
    This class generates a threadmanager object.
    This manager can start daemon threads and signal when the thread is finished.
    This class is only used to spawn calls to the C++/CUDA backend inside a daemon thread so that Python keeps control over the process.
    This allows users to Ctrl-c a running calculation in C++/CUDA from Python.
    """

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
