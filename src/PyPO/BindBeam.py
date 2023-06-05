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
# These bindings are concerned with beam generation for the ray-tracer and the physical optics.

##
# Load the PyPObeam shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadBeamlib():
    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.dylib"))

    lib.makeRTframe.argtypes = [RTDict, ctypes.POINTER(cframe)]
    lib.makeRTframe.restype = None
    
    lib.makeGRTframe.argtypes = [GRTDict, ctypes.POINTER(cframe)]
    lib.makeGRTframe.restype = None

    lib.makeGauss.argtypes = [GPODict, reflparams, ctypes.POINTER(c2Bundle), ctypes.POINTER(c2Bundle)]
    lib.makeGauss.restype = None

    lib.makeScalarGauss.argtypes = [ScalarGPODict, reflparams, ctypes.POINTER(arrC1)]
    lib.makeScalarGauss.restype = None
    
    lib.calcCurrents.argtypes = [ctypes.POINTER(c2Bundle), ctypes.POINTER(c2Bundle),
                                reflparams, ctypes.c_int]
    lib.calcCurrents.restype = None

    return lib

##
# Generate a tubular ray-trace frame.
# The tube consists of annular rings of rays and can be given opening angles and radii.
#
# @param RTDict_py A filled TubeRTDict.
#
# @returns out A frame object containing the ray-trace frame.
#
# @see TubeRTDict
# @see frame
def makeRTframe(RTDict_py):
    lib = loadBeamlib()

    nTot = 1 + RTDict_py["nRays"] * 4 * RTDict_py["nRing"]

    RTDict_c = RTDict()
    res = cframe()

    allocate_cframe(res, nTot, ctypes.c_double)
    allfill_RTDict(RTDict_c, RTDict_py, ctypes.c_double)

    lib.makeRTframe(RTDict_c, ctypes.byref(res))

    shape = (nTot,)
    out = frameToObj(res, np_t=np.float64, shape=shape)

    return out

##
# Generate a Gaussian ray-trace frame.
# The Gaussian ray-trace frame has positions and directions chosen from a Gaussian distribution..
#
# @param grdict_py A filled GRTDict.
#
# @returns out A frame object containing the ray-trace frame.
# 
# @see GRTDict
# @see frame
def makeGRTframe(grdict_py):
    lib = loadBeamlib()
    mgr = TManager.Manager(Config.context)

    nTot = grdict_py["nRays"]

    c_grdict = GRTDict()
    res = cframe()
    
    allocate_cframe(res, nTot, ctypes.c_double)
    allfill_GRTDict(c_grdict, grdict_py, ctypes.c_double)
    
    args = [c_grdict, ctypes.byref(res)]
    mgr.new_sthread(target=lib.makeGRTframe, args=args)
    
    shape = (nTot,)
    out = frameToObj(res, np_t=np.float64, shape=shape)

    return out

##
# Generate a polarised Gaussian beam.
# The beam is always defined parallel to the x, y plane. The z-coordinate can be adjusted.
# In order to tilt the beam, you have to tilt the underlying plane AFTER defining the beam on it.
#
# @param gdict_py A GPODict dictionary containing relevant Gaussian beam parameters.
# @param source A reflDict dictionary describing the plane on which the Gaussian is defined.
#
# @returns out_field Field object containing the electromagnetic fields associated with the Gaussian.
# @returns out_current Current object containing the electromagnetic currents associated with the Gaussian.
#
# @see GPODict
# @see reflDict
# @see fields
# @see currents
def makeGauss(gdict_py, source):
    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_gdict = GPODict()
    c_source = reflparams()

    allfill_GPODict(c_gdict, gdict_py, ctypes.c_double)
    allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = c2Bundle()
    res_current = c2Bundle()
    allocate_c2Bundle(res_field, source_size, ctypes.c_double)
    allocate_c2Bundle(res_current, source_size, ctypes.c_double)

    lib.makeGauss(c_gdict, c_source, ctypes.byref(res_field), ctypes.byref(res_current))

    out_field = c2BundleToObj(res_field, shape=source_shape, obj_t='fields', np_t=np.float64)
    out_current = c2BundleToObj(res_current, shape=source_shape, obj_t='currents', np_t=np.float64)

    return out_field, out_current

##
# Generate a scalar Gaussian beam.
# The beam is always defined parallel to the x, y plane. The z-coordinate can be adjusted.
# In order to tilt the beam, you have to tilt the underlying plane AFTER defining the beam on it.
#
# @param gdict_py A GPODict dictionary containing relevant scalar Gaussian beam parameters.
# @param source A reflDict dictionary describing the plane on which the scalar Gaussian is defined.
#
# @returns out_field Scalarfield object containing the electric scalar field associated with the Gaussian.
#
# @see GPODict
# @see reflDict
# @see fields
def makeScalarGauss(gdict_py, source):
    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_gdict = ScalarGPODict()
    c_source = reflparams()
    allfill_SGPODict(c_gdict, gdict_py, ctypes.c_double)
    allfill_reflparams(c_source, source, ctypes.c_double)
    
    res_field = arrC1()
    allocate_arrC1(res_field, source_size, ctypes.c_double)
    lib.makeScalarGauss(c_gdict, c_source, ctypes.byref(res_field))
    
    out_field = arrC1ToObj(res_field, shape=source_shape, np_t=np.float64)

    return out_field

##
# Calculate electromagnetic currents from electromagnetic field.
#
# @param fields Fields object containing electromagnetic fields.
# @param source A reflDict dictionary describing the plane on which the Gaussian is defined.
# @param mode Whether to assume plane is perfect electrical conductor ('PEC'), magnetic conductor ('PMC') or no assumptions ('full').
#
# @returns out_current Currents object containing the currents calculated on source.
#
# @see fields
# @see currents
def calcCurrents(fields, source, mode):
    lib = loadBeamlib()
    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_source = reflparams()

    allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = c2Bundle()
    res_current = c2Bundle()
    allfill_c2Bundle(res_field, fields, fields.size, ctypes.c_double)
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
