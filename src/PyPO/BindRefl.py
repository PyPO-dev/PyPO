"""!
@file
Bindings for the ctypes interface for PyPO. 
These bindings are concerned with generation of reflector grids from dictionaries.
"""

import ctypes
import numpy as np
import os
import sys
import pathlib

import PyPO.BindUtils as BUtils
import PyPO.Structs as PStructs

def loadRefllib():
    """!
    Load the PyPOrefl shared library. Will detect the operating system and link the library accordingly.

    @returns lib The ctypes library containing the C/C++ functions.
    """

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpyporefl.dylib"))

    return lib

def generateGrid(reflparams_py, transform=True, spheric=True): 
    """!
    Double precision function for generating reflector grids.
    This is the function called by the CPU PyPO bindings.
    Also, when called from System, this is the binding that is called.

    @param reflparams_py A reflDict dictionary.
    @param transform Whether to generate the grid in nominal configuration or to apply transformation matrix.
    @param spheric Convert Az-El co-ordinates to spherical (far-field only).

    @returns grids A reflGrids object. 
    """

    lib = loadRefllib()

    size = reflparams_py["gridsize"][0] * reflparams_py["gridsize"][1]

    inp = PStructs.reflparams()
    res = PStructs.reflcontainer()
    

    BUtils.allfill_reflparams(inp, reflparams_py, ctypes.c_double)
    BUtils.allocate_reflcontainer(res, size, ctypes.c_double)

    lib.generateGrid.argtypes = [PStructs.reflparams, ctypes.POINTER(PStructs.reflcontainer),
                                ctypes.c_bool, ctypes.c_bool]
    lib.generateGrid.restype = None

    lib.generateGrid(inp, ctypes.byref(res), transform, spheric)
    

    grids = BUtils.creflToObj(res, reflparams_py["gridsize"], np.float64)
    

    return grids
