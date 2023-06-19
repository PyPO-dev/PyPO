import ctypes
import numpy as np
import os
import pathlib
import traceback

import PyPO.BindUtils as BUtils
import PyPO.Structs as PStructs
import PyPO.Config as Config
import PyPO.Threadmgr as TManager

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with beam generation for the ray-tracer and the physical optics.

def loadBeamlib():
    """!
    Load the PyPObeam shared library. Will detect the operating system and link the library accordingly.

    @returns lib The ctypes library containing the C/C++ functions.
    """

    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypobeam.dylib"))

    lib.makeRTframe.argtypes = [PStructs.RTDict, ctypes.POINTER(PStructs.cframe)]
    lib.makeRTframe.restype = None
    
    lib.makeGRTframe.argtypes = [PStructs.GRTDict, ctypes.POINTER(PStructs.cframe)]
    lib.makeGRTframe.restype = None

    lib.makeGauss.argtypes = [PStructs.GPODict, PStructs.reflparams, ctypes.POINTER(PStructs.c2Bundle), ctypes.POINTER(PStructs.c2Bundle)]
    lib.makeGauss.restype = None

    lib.makeScalarGauss.argtypes = [PStructs.ScalarGPODict, PStructs.reflparams, ctypes.POINTER(PStructs.arrC1)]
    lib.makeScalarGauss.restype = None
    
    lib.calcCurrents.argtypes = [ctypes.POINTER(PStructs.c2Bundle), ctypes.POINTER(PStructs.c2Bundle),
                                PStructs.reflparams, ctypes.c_int]
    lib.calcCurrents.restype = None

    return lib

def makeRTframe(RTDict_py):
    """!
    Generate a tubular ray-trace frame.
    The tube consists of annular rings of rays and can be given opening angles and radii.

    @param RTDict_py A filled TubeRTDict.

    @returns out A frame object containing the ray-trace frame.

    @see TubeRTDict
    @see frame
    """

    lib = loadBeamlib()

    nTot = 1 + RTDict_py["nRays"] * 4 * RTDict_py["nRing"]

    RTDict_c = PStructs.RTDict()
    res = PStructs.cframe()

    BUtils.allocate_cframe(res, nTot, ctypes.c_double)
    BUtils.allfill_RTDict(RTDict_c, RTDict_py, ctypes.c_double)

    lib.makeRTframe(RTDict_c, ctypes.byref(res))

    shape = (nTot,)
    out = BUtils.frameToObj(res, np_t=np.float64, shape=shape)

    return out

def makeGRTframe(grdict_py):
    """!
    Generate a Gaussian ray-trace frame.
    The Gaussian ray-trace frame has positions and directions chosen from a Gaussian distribution..

    @param grdict_py A filled GRTDict.

    @returns out A frame object containing the ray-trace frame.

    @see GRTDict
    @see frame
    """

    lib = loadBeamlib()
    mgr = TManager.Manager(Config.context)

    nTot = grdict_py["nRays"]

    c_grdict = PStructs.GRTDict()
    res = PStructs.cframe()
   

    BUtils.allocate_cframe(res, nTot, ctypes.c_double)
    BUtils.allfill_GRTDict(c_grdict, grdict_py, ctypes.c_double)
    
    args = [c_grdict, ctypes.byref(res)]
    mgr.new_sthread(target=lib.makeGRTframe, args=args)
    
    shape = (nTot,)
    out = BUtils.frameToObj(res, np_t=np.float64, shape=shape)

    return out

def makeGauss(gdict_py, source):
    """!
    Generate a polarised Gaussian beam.
    The beam is always defined parallel to the x, y plane. The z-coordinate can be adjusted.
    In order to tilt the beam, you have to tilt the underlying plane AFTER defining the beam on it.

    @param gdict_py A GPODict dictionary containing relevant Gaussian beam parameters.
    @param source A reflDict dictionary describing the plane on which the Gaussian is defined.

    @returns out_field Field object containing the electromagnetic fields associated with the Gaussian.
    @returns out_current Current object containing the electromagnetic currents associated with the Gaussian.

    @see GPODict
    @see reflDict
    @see fields
    @see currents
    """

    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_gdict = PStructs.GPODict()
    c_source = PStructs.reflparams()

    BUtils.allfill_GPODict(c_gdict, gdict_py, ctypes.c_double)
    BUtils.allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = PStructs.c2Bundle()
    res_current = PStructs.c2Bundle()
    BUtils.allocate_c2Bundle(res_field, source_size, ctypes.c_double)
    BUtils.allocate_c2Bundle(res_current, source_size, ctypes.c_double)

    lib.makeGauss(c_gdict, c_source, ctypes.byref(res_field), ctypes.byref(res_current))

    out_field = BUtils.c2BundleToObj(res_field, shape=source_shape, obj_t='fields', np_t=np.float64)
    out_current = BUtils.c2BundleToObj(res_current, shape=source_shape, obj_t='currents', np_t=np.float64)

    return out_field, out_current

def makeScalarGauss(gdict_py, source):
    """!
    Generate a scalar Gaussian beam.
    The beam is always defined parallel to the x, y plane. The z-coordinate can be adjusted.
    In order to tilt the beam, you have to tilt the underlying plane AFTER defining the beam on it.

    @param gdict_py A GPODict dictionary containing relevant scalar Gaussian beam parameters.
    @param source A reflDict dictionary describing the plane on which the scalar Gaussian is defined.

    @returns out_field Scalarfield object containing the electric scalar field associated with the Gaussian.

    @see GPODict
    @see reflDict
    @see fields
    """

    lib = loadBeamlib()

    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_gdict = PStructs.ScalarGPODict()
    c_source = PStructs.reflparams()
    BUtils.allfill_SGPODict(c_gdict, gdict_py, ctypes.c_double)
    BUtils.allfill_reflparams(c_source, source, ctypes.c_double)
    
    res_field = PStructs.arrC1()
    BUtils.allocate_arrC1(res_field, source_size, ctypes.c_double)
    lib.makeScalarGauss(c_gdict, c_source, ctypes.byref(res_field))
    
    out_field = BUtils.arrC1ToObj(res_field, shape=source_shape, np_t=np.float64)

    return out_field

def calcCurrents(fields, source, mode):
    """!
    Calculate electromagnetic currents from electromagnetic field.

    @param fields Fields object containing electromagnetic fields.
    @param source A reflDict dictionary describing the plane on which the Gaussian is defined.
    @param mode Whether to assume plane is perfect electrical conductor ('PEC'), magnetic conductor ('PMC') or no assumptions ('full').

    @returns out_current Currents object containing the currents calculated on source.

    @see fields
    @see currents
    """

    lib = loadBeamlib()
    source_shape = (source["gridsize"][0], source["gridsize"][1])
    source_size = source["gridsize"][0] * source["gridsize"][1]

    c_source = PStructs.reflparams()

    BUtils.allfill_reflparams(c_source, source, ctypes.c_double)

    res_field = PStructs.c2Bundle()
    res_current = PStructs.c2Bundle()
    BUtils.allfill_c2Bundle(res_field, fields, fields.size, ctypes.c_double)
    BUtils.allocate_c2Bundle(res_current, source_size, ctypes.c_double)

    if mode == "full":
        mode = 0

    elif mode == "PMC":
        mode = 1

    elif mode == "PEC":
        mode = 2

    mode = ctypes.c_int(mode)

    lib.calcCurrents(ctypes.byref(res_field), ctypes.byref(res_current),
                    c_source, mode)

    out_current = BUtils.c2BundleToObj(res_current, shape=source_shape, obj_t='currents', np_t=np.float64)

    return out_current
