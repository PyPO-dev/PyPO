import ctypes
import numpy as np
import os
import pathlib
from PyPO.BindUtils import *
from PyPO.Structs import *
from PyPO.PyPOTypes import *
import PyPO.Config as Config
import PyPO.Threadmgr as TManager

##
# @file
# Bindings for the ctypes interface for PyPO. 
# These bindings are concerned with propagations for the ray-tracer and the physical optics on the CPU.

##
# Load the PyPOcpu shared library. Will detect the operating system and link the library accordingly.
#
# @returns lib The ctypes library containing the C/C++ functions.
def loadCPUlib():
    path_cur = pathlib.Path(__file__).parent.resolve()
    try:
        lib = ctypes.CDLL(os.path.join(path_cur, "libpypocpu.dll"))
    except:
        try:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypocpu.so"))
        except:
            lib = ctypes.CDLL(os.path.join(path_cur, "libpypocpu.dylib"))

    lib.propagateToGrid_JM.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JM.restype = None

    lib.propagateToGrid_EH.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EH.restype = None

    lib.propagateToGrid_JMEH.argtypes = [ctypes.POINTER(c4Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JMEH.restype = None

    lib.propagateToGrid_EHP.argtypes = [ctypes.POINTER(c2rBundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EHP.restype = None

    lib.propagateToGrid_scalar.argtypes = [ctypes.POINTER(arrC1), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(arrC1),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_scalar.restype = None

    lib.propagateToFarField.argtypes = [ctypes.POINTER(c2Bundle), reflparams, reflparams,
                                        ctypes.POINTER(reflcontainer), ctypes.POINTER(reflcontainer),
                                        ctypes.POINTER(c2Bundle),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToFarField.restype = None

    lib.propagateRays.argtypes = [reflparams, ctypes.POINTER(cframe),
                                ctypes.POINTER(cframe), ctypes.c_int, ctypes.c_double, ctypes.c_double]

    lib.propagateRays.restype = None

    return lib

##
# Perform a PO propagation on the CPU.
# Note that the calculations are always done in double precision for the CPU.
# Depending on the 'mode' parameter in the runPODict, this function returns different objects.
# Please see the dictionary templates for an overview.
#
# @param source A reflDict dictionary of the surface on which the source currents/scalarfields are defined.
# @param target A reflDict dictionary of the target surface on which the results are calculated.
# @param runPODict A runPODict dictionary containing the relevant propagation parameters.
#
# @see runPODict
def PyPO_CPUd(source, target, runPODict):
    lib = loadCPUlib()
    mgr = TManager.Manager(Config.context)

    # Create structs with pointers for source and target
    csp = reflparams()
    ctp = reflparams()

    cs = reflcontainer()
    ct = reflcontainer()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_double)
    allfill_reflparams(ctp, target, ctypes.c_double)

    allocate_reflcontainer(cs, gs, ctypes.c_double)
    allocate_reflcontainer(ct, gt, ctypes.c_double)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    if runPODict["exp"] == "fwd":
        exp_prop = -1

    elif runPODict["exp"] == "bwd":
        exp_prop = 1

    k           = ctypes.c_double(runPODict["k"])
    nThreads    = ctypes.c_int(runPODict["nThreads"])
    epsilon     = ctypes.c_double(runPODict["epsilon"])
    t_direction = ctypes.c_double(exp_prop)
    
    if runPODict["mode"] == "scalar":
        c_field = arrC1()
        sfieldConv(runPODict["s_scalarfield"], c_field, gs, ctypes.c_double)
        args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
                ctypes.byref(c_field), k, nThreads, epsilon,
                t_direction]

    else:
        c_currents = c2Bundle()
        allfill_c2Bundle(c_currents, runPODict["s_current"], gs, ctypes.c_double)
        args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
                ctypes.byref(c_currents), k, nThreads, epsilon,
                t_direction]



    if runPODict["mode"] == "JM":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToGrid_JM, args=args)

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif runPODict["mode"] == "EH":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToGrid_EH, args=args)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif runPODict["mode"] == "JMEH":
        res = c4Bundle()
        args.insert(0, res)

        allocate_c4Bundle(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToGrid_JMEH, args=args)

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif runPODict["mode"] == "EHP":
        res = c2rBundle()
        args.insert(0, res)

        allocate_c2rBundle(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToGrid_EHP, args=args)

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif runPODict["mode"] == "scalar":
        res = arrC1()
        args.insert(0, res)

        allocate_arrC1(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToGrid_scalar, args=args)

        # Unpack filled struct
        S = arrC1ToObj(res, shape=target_shape, np_t=np.float64)

        return S

    elif runPODict["mode"] == "FF":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        mgr.new_sthread(target=lib.propagateToFarField, args=args)

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

##
# Perform an RT propagation on the CPU.
# Note that the calculations are always done in double precision for the CPU.
#
# @param runRTDict A runRTDict dictionary containing the relevant propagation parameters.
#
# @see runRTDict
def RT_CPUd(runRTDict):
    lib = loadCPUlib()
    mgr = TManager.Manager(Config.context)

    inp = cframe()
    res = cframe()

    allocate_cframe(res, runRTDict["fr_in"].size, ctypes.c_double)
    allfill_cframe(inp, runRTDict["fr_in"], runRTDict["fr_in"].size, ctypes.c_double)

    ctp = reflparams()
    allfill_reflparams(ctp, runRTDict["t_name"], ctypes.c_double)

    nThreads    = ctypes.c_int(runRTDict["nThreads"])
    tol         = ctypes.c_double(runRTDict["tol"])
    t0          = ctypes.c_double(runRTDict["t0"])

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
                        nThreads, tol, t0]
    
    mgr.new_sthread(target=lib.propagateRays, args=args)

    shape = (runRTDict["fr_in"].size,)
    fr_out = frameToObj(res, np_t=np.float64, shape=shape)
    return fr_out
