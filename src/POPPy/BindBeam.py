import ctypes
import math
import numpy as np

from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

#############################################################################
#                                                                           #
#           List of bindings for the beam init interface of POPPy.          #
#                                                                           #
#############################################################################

def loadBeamlib():
    lib = ctypes.CDLL('./src/C++/libpoppybeam.so')

    lib.makeRTframe.argtypes = [RTDict, ctypes.POINTER(cframe)]

    lib.makeRTframe.restype = None

    return lib

def makeRTframe(rdict_py):
    lib = loadBeamlib()

    nTot = 1 + rdict_py["nRays"] * 4 * rdict_py["nRing"]

    c_rdict = RTDict()
    res = cframe()

    allocate_cframe(res, nTot, ctypes.c_double)
    allfill_RTDict(c_rdict, rdict_py, ctypes.c_double)

    lib.makeRTframe(c_rdict, ctypes.byref(res))

    shape = (nTot,)
    out = frameToObj(res, np_t=np.float64, shape=shape)

    return out
