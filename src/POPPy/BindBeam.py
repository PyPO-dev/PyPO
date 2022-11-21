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
    lib = ctypes.CDLL('./src/libpoppybeam.so')

    lib.makeRTframe.argtypes = [RTDict, ctypes.POINTER(cframe)]
    lib.makeRTframe.restype = None

    lib.makeGauss.argtypes = [GDict, reflparams, ctypes.POINTER(c2Bundle), ctypes.POINTER(c2Bundle)]
    lib.makeGauss.restype = None

    lib.calcCurrents.argtypes = [ctypes.POINTER(c2Bundle), ctypes.POINTER(c2Bundle),
                                reflparams, ctypes.c_int]
    lib.calcCurrents.restype = None

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

def makeGauss(gdict_py, source):
    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_gdict = GDict()
    c_source = reflparams()

    allfill_GDict(c_gdict, gdict_py, ctypes.c_double)
    allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = c2Bundle()
    res_current = c2Bundle()
    allocate_c2Bundle(res_field, source_size, ctypes.c_double)
    allocate_c2Bundle(res_current, source_size, ctypes.c_double)

    lib.makeGauss(c_gdict, c_source, ctypes.byref(res_field), ctypes.byref(res_current))

    out_field = c2BundleToObj(res_field, shape=source_shape, obj_t='fields', np_t=np.float64)
    out_current = c2BundleToObj(res_current, shape=source_shape, obj_t='currents', np_t=np.float64)

    return out_field, out_current

def calcCurrents(fields, source, mode):
    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_source = reflparams()

    allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = c2Bundle()
    res_current = c2Bundle()
    fieldConv(fields, res_field, source_size, ctypes.c_double)
    allocate_c2Bundle(res_current, source_size, ctypes.c_double)

    if mode == "full":
        mode = 0

    elif mode == "PMC":
        mode = 1

    elif mode == "PEC":
        mode = 2

    mode = ctypes.c_int(mode)

    lib.calcCurrents(ctypes.byref(res_field), ctypes.byref(res_current),
                    c_source, mode)

    out_current = c2BundleToObj(res_current, shape=source_shape, obj_t='currents', np_t=np.float64)

    return out_current
