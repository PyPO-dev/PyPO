import numpy as np

##
# @file
# Set global world parameters for PyPO.

ORIGIN      = np.zeros(3)
XAX       = np.array([1, 0, 0])
YAX       = np.array([0, 1, 0])
ZAX       = np.array([0, 0, 1])

IAX       = ZAX

INITM       = np.eye(4)

def writeSettings():
    pass
