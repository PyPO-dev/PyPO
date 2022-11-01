import ctypes
import numpy as np

from src.POPPy.BindUtils import allfill_reflparams, allocate_reflcontainer, creflToObj
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

#############################################################################
#                                                                           #
#           List of bindings for the reflector interface of POPPy.          #
#                                                                           #
#############################################################################

def loadRefllib():
    lib = ctypes.CDLL('./src/C++/libpoppyrefl.so')
    return lib

#### DOUBLE PRECISION
def generateGrid(reflparams_py):
    lib = loadRefllib()
    size = reflparams_py["gridsize"][0] * reflparams_py["gridsize"][1]

    inp = reflparams()
    res = reflcontainer()

    allfill_reflparams(inp, reflparams_py, ctypes.c_double)
    allocate_reflcontainer(res, size, ctypes.c_double)

    lib.generateGrid.argtypes = [reflparams, ctypes.POINTER(reflcontainer)]
    lib.generateGrid.restype = None

    lib.generateGrid(inp, ctypes.byref(res))
    grids = creflToObj(res, reflparams_py["gridsize"], np.float64)

    return grids

#### SINGLE PRECISION
def generateGridf(reflparams_py):
    lib = loadRefllib()
    size = reflparams_py["gridsize"][0] * reflparams_py["gridsize"][1]

    inp = reflparamsf()
    res = reflcontainerf()

    allfill_reflparams(inp, reflparams_py, ctypes.c_float)
    allocate_reflcontainer(res, size, ctypes.c_float)

    lib.generateGridf.argtypes = [reflparamsf, ctypes.POINTER(reflcontainerf)]
    lib.generateGridf.restype = None

    lib.generateGridf(inp, ctypes.byref(res))

    grids = creflToObj(res, reflparams_py["gridsize"], np.float32)

    return grids

if __name__ == "__main__":
    print("Bindings for POPPy reflectors.")
