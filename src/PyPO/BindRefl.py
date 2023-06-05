import ctypes
import numpy as np
import os
import sys
import pathlib

from PyPO.BindUtils import allfill_reflparams, allocate_reflcontainer, creflToObj
from PyPO.Structs import *
from PyPO.PyPOTypes import *

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with generation of reflector grids from dictionaries.

##
# Load the PyPOgpu shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadRefllib():
    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.dylib"))

    return lib

##
# Double precision function for generating reflector grids.
# This is the function called by the CPU PyPO bindings.
# Also, when called from System, this is the binding that is called.
# 
# @param reflparams_py A reflDict dictionary.
# @param transform Whether to generate the grid in nominal configuration or to apply transformation matrix.
# @param spheric Convert Az-El co-ordinates to spherical (far-field only).
#
# @returns grids A reflGrids object. 
def generateGrid(reflparams_py, transform=True, spheric=True):
    
    lib = loadRefllib()
    

    size = reflparams_py["gridsize"][0] * reflparams_py["gridsize"][1]

    inp = reflparams()
    res = reflcontainer()
    

    allfill_reflparams(inp, reflparams_py, ctypes.c_double)
    allocate_reflcontainer(res, size, ctypes.c_double)

    lib.generateGrid.argtypes = [reflparams, ctypes.POINTER(reflcontainer),
                                ctypes.c_bool, ctypes.c_bool]
    lib.generateGrid.restype = None

    lib.generateGrid(inp, ctypes.byref(res), transform, spheric)
    

    grids = creflToObj(res, reflparams_py["gridsize"], np.float64)
    

    return grids

##
# Single precision function for generating reflector grids.
# This is the function called by the GPU PyPO bindings.
# 
# @param reflparams_py A reflDict dictionary.
# @param transform Whether to generate the grid in nominal configuration or to apply transformation matrix.
# @param spheric Convert Az-El co-ordinates to spherical (far-field only).
#
# @returns grids A reflGrids object. 
def generateGridf(reflparams_py, transform, spheric):
    lib = loadRefllib()
    size = reflparams_py["gridsize"][0] * reflparams_py["gridsize"][1]

    inp = reflparamsf()
    res = reflcontainerf()

    allfill_reflparams(inp, reflparams_py, ctypes.c_float)
    allocate_reflcontainer(res, size, ctypes.c_float)

    lib.generateGridf.argtypes = [reflparamsf, ctypes.POINTER(reflcontainerf),
                                ctypes.c_bool, ctypes.c_bool]
    lib.generateGridf.restype = None

    lib.generateGridf(inp, ctypes.byref(res), transform, spheric)

    grids = creflToObj(res, reflparams_py["gridsize"], np.float32)

    return grids

if __name__ == "__main__":
    print("Bindings for PyPO reflectors.")
