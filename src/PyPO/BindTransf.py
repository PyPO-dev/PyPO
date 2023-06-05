import ctypes
import numpy as np
import os
import pathlib
import traceback

from PyPO.BindUtils import *
from PyPO.Structs import *
from PyPO.PyPOTypes import *
import PyPO.Config as Config
import PyPO.Threadmgr as TManager

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with transforming frame and fields/currents.

##
# Load the pypotransf shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadTransflib():
    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpypotransf.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypotransf.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypotransf.dylib"))

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
   
    nTot = ctypes.c_int(obj.size)

    lib.transformFields(ctypes.byref(res), c_mat, nTot)

    out = c2BundleToObj(res, shape=obj.shape, obj_t=obj_type, np_t=np.float64)
    
    out.setMeta(obj.surf, obj.k)
    return out

