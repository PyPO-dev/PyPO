"""!
@file
Bindings for the ctypes interface for PyPO. 
These bindings are concerned with transforming frame and fields/currents.
"""

import ctypes
import numpy as np
import os
import pathlib
import traceback

import PyPO.BindUtils as BUtils
import PyPO.Structs as PStructs
import PyPO.Config as Config
import PyPO.Threadmgr as TManager

def loadTransflib():
    """!
    Load the pypotransf shared library. Will detect the operating system and link the library accordingly.

    @returns lib The ctypes library containing the C/C++ functions.
    """

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "pypotransf.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypotransf.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypotransf.dylib"))

    lib.transformRays.argtypes = [ctypes.POINTER(PStructs.cframe), ctypes.POINTER(ctypes.c_double)]
    lib.transformRays.restype = None

    lib.transformFields.argtypes = [ctypes.POINTER(PStructs.c2Bundle), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.transformFields.restype = None
    
    return lib

def transformRays(fr):
    """!
    Transform a frame of rays. 

    @param fr A frame object.

    @see frame
    """

    lib = loadTransflib()

    res = PStructs.cframe()

    BUtils.allfill_cframe(res, fr, fr.size, ctypes.c_double)
    c_mat = BUtils.allfill_mat4D(fr.transf, ctypes.c_double)

    lib.transformRays(ctypes.byref(res), c_mat)

    shape = (fr.size,)
    out = BUtils.frameToObj(res, np_t=np.float64, shape=shape)
    
    out.setMeta(fr.pos, fr.ori, fr.transf)
    out.snapshots = fr.snapshots
    return out

def transformPO(obj, transf):
    """!
    Transform a frame of rays. 

    @param fr A frame object.

    @see frame
    """

    lib = loadTransflib()

    res = PStructs.c2Bundle()
    BUtils.allfill_c2Bundle(res, obj, obj.size, ctypes.c_double)
    c_mat = BUtils.allfill_mat4D(transf, ctypes.c_double)

    obj_type = "fields"
    
    if obj.type == "JM":
        obj_type = "currents"
   
    nTot = ctypes.c_int(obj.size)

    lib.transformFields(ctypes.byref(res), c_mat, nTot)

    out = BUtils.c2BundleToObj(res, shape=obj.shape, obj_t=obj_type, np_t=np.float64)
    
    out.setMeta(obj.surf, obj.k)
    return out

