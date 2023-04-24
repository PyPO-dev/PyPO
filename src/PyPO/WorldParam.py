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

def ORIGIN():
    return np.zeros(3)

def XAX():
    return np.array([1, 0, 0])

def YAX():
    return np.array([0, 1, 0])

def ZAX():
    return np.array([0, 0, 1])

def IAX():
    return ZAX()

def INITM():
    return np.eye(4)

def writeSettings():
    pass
