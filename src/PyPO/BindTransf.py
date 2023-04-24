import ctypes
import numpy as np
import os
import sys
import time
import pathlib
from src.PyPO.BindUtils import *
from src.PyPO.Structs import *
from src.PyPO.PyPOTypes import *
import src.PyPO.Config as Config
import src.PyPO.Threadmgr as TManager

import threading

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with transforming frame and fields/currents.

##
# Load the pypotransf shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadTransflib():
    try:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build/Debug"
        lib = ctypes.CDLL(str(LD_PATH/"pypotransf.dll"))
    except:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build"
        try:
            lib = ctypes.CDLL(LD_PATH/"libpypotransf.so")
        except:
            lib = ctypes.CDLL(LD_PATH/"libpypotransf.dylib")

    lib.transformRays.argtypes = [ctypes.POINTER(cframe), ctypes.POINTER(ctypes.c_double)]
    lib.transformRays.restype = None

    lib.transformFields.argtypes = [ctypes.POINTER(c2Bundle), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.transformFields.restype = None
    
    return lib

##
# Transform a frame of rays. 
#
# @param fr A frame object.
#
# @see frame
def transformRays(fr):
    lib = loadTransflib()

    res = cframe()

    allfill_cframe(res, fr, fr.size, ctypes.c_double)
    c_mat = allfill_mat4D(fr.transf, ctypes.c_double)

    lib.transformRays(ctypes.byref(res), c_mat)

    shape = (fr.size,)
    out = frameToObj(res, np_t=np.float64, shape=shape)
    
    out.setMeta(fr.pos, fr.ori, fr.transf)
    out.snapshots = fr.snapshots
    return out

##
# Transform a frame of rays. 
#
# @param fr A frame object.
#
# @see frame
def transformPO(obj, transf):
    lib = loadTransflib()

    res = c2Bundle()
    allfill_c2Bundle(res, obj, obj.size, ctypes.c_double)
    c_mat = allfill_mat4D(transf, ctypes.c_double)

    obj_type = "fields"
    
    if obj.type == "JM":
        obj_type = "currents"
        #currentConv(obj, res, obj.size, ctypes.c_double)
    
        #fieldConv(obj, res, obj.size, ctypes.c_double)
   
    nTot = ctypes.c_int(obj.size)

    lib.transformFields(ctypes.byref(res), c_mat, nTot)

    out = c2BundleToObj(res, shape=obj.shape, obj_t=obj_type, np_t=np.float64)
    
    out.setMeta(obj.surf, obj.k)
    return out

