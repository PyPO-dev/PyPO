import ctypes
import numpy as np

from src.POPPy.BindUtils import *
from src.POPPy.Structs import *
from src.POPPy.POPPyTypes import *

import sys

#############################################################################
#                                                                           #
#              List of bindings for the CPU interface of POPPy.             #
#                                                                           #
#############################################################################

def loadCPUlib():
    lib = ctypes.CDLL('./src/C++/libpoppycpu.so')
    return lib

# WRAPPER FUNCTIONS DOUBLE PREC
def calcJM_CPUd(source, target, currents, k, epsilon, t_direction, nThreads):
    lib = loadCPUlib()

    # Create structs with pointers for source and target
    csp = reflparamsf()
    ctp = reflparamsf()

    cs = reflcontainerf()
    ct = reflcontainerf()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_double)
    allfill_reflparams(ctp, target, ctypes.c_double)

    allocate_reflcontainer(cs, gs, ctypes.c_double)
    allocate_reflcontainer(ct, gt, ctypes.c_double)

    # Create structure for input currents c2Bundle
    c_currents = c2Bundlef()

    # Copy content of currents into c_currents
    currentConv(currents, c_currents, gs, ctypes.c_double)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    # Define arg and return types
    lib.propagateToGrid_JM.argtypes = [ctypes.POINTER(c2Bundle), reflparamsf, reflparamsf,
                                        ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                        ctypes.POINTER(c2Bundlef),ctypes.c_double, ctypes.c_int,
                                        ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JM.restype = None

    k           = ctypes.c_double(k)
    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundle()
    allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_double)

    lib.propagateToGrid_JM(ctypes.byref(res), csp, ctp,
                            ctypes.byref(cs), ctypes.byref(ct),
                            ctypes.byref(c_currents), k, nThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    JM = c2BundleToObj(res, shape=target_shape, obj_t='currents')

    return JM

def calcEH_CPUd(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGrid_EH.argtypes = [ctypes.POINTER(c2Bundle),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EH.restype = None

    xyzs, xyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_double, normals=False)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_double(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundle()

    lib.propagateToGrid_EH(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

    return EH

def calcJMEH_CPUd(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGrid_JMEH.argtypes = [ctypes.POINTER(c4Bundle),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_JMEH.restype = None

    xyzs, xyzt, nxyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_double)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_double(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = c4Bundle()

    lib.propagateToGrid_JMEH(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            nxyzt[0], nxyzt[1], nxyzt[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    JM, EH = c4BundleToObj(res, shape=target_shape)

    return JM, EH

def calcEHP_CPUd(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGrid_EHP.argtypes = [ctypes.POINTER(c2rBundle),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_EHP.restype = None

    xyzs, xyzt, nxyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_double)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_double(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = c2rBundle()

    lib.propagateToGrid_EHP(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            nxyzt[0], nxyzt[1], nxyzt[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH, Pr = c2rBundleToObj(res, shape=target_shape)

    return EH, Pr

def calcScalar_CPUd(source, target, field, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGrid_scalar.argtypes = [ctypes.POINTER(arrC1),
                                       ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double]

    lib.propagateToGrid_scalar.restype = None

    xyzs, xyzt, area, rEs, iEs = extractorScalar(source, target, field, ctypes.c_double)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_double(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = arrC1()

    lib.propagateToGrid_scalar(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            rEs, iEs,
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    E = arrC1ToObj(res, shape=target_shape)

    return E

def calcFF_CPUd(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    print(currents.My)

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToFarField.argtypes = [ctypes.POINTER(c2Bundle),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                       ctypes.c_double, ctypes.c_double]

    lib.propagateToFarField.restype = None

    xyzs, xyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_double, normals=False)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_double(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_double(epsilon)
    t_direction = ctypes.c_double(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundle()

    lib.propagateToFarField(ctypes.byref(res),
                            xyzt[0], xyzt[1],
                            xyzs[0], xyzs[1], xyzs[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

    return EH

# WRAPPER FUNCTIONS SINGLE PREC
def calcJM_CPUf(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGridf_JM.argtypes = [ctypes.POINTER(c2Bundlef),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToGridf_JM.restype = None

    xyzs, xyzt, nxyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_float)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundlef()

    lib.propagateToGridf_JM(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            nxyzt[0], nxyzt[1], nxyzt[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    JM = c2BundleToObj(res, shape=target_shape, obj_t='currents')

    return JM

def calcEH_CPUf(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGridf_EH.argtypes = [ctypes.POINTER(c2Bundlef),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToGridf_EH.restype = None

    xyzs, xyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_float, normals=False)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundlef()

    lib.propagateToGridf_EH(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

    return EH

def calcJMEH_CPUf(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGridf_JMEH.argtypes = [ctypes.POINTER(c4Bundlef),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToGridf_JMEH.restype = None

    xyzs, xyzt, nxyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_float)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = c4Bundlef()

    lib.propagateToGridf_JMEH(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            nxyzt[0], nxyzt[1], nxyzt[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    JM, EH = c4BundleToObj(res, shape=target_shape)

    return JM, EH

def calcEHP_CPUf(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGridf_EHP.argtypes = [ctypes.POINTER(c2rBundlef),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToGridf_EHP.restype = None

    xyzs, xyzt, nxyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_float)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = c2rBundlef()

    lib.propagateToGridf_EHP(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            nxyzt[0], nxyzt[1], nxyzt[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH, Pr = c2rBundleToObj(res, shape=target_shape)

    return EH, Pr

def calcScalar_CPUf(source, target, field, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToGridf_scalar.argtypes = [ctypes.POINTER(arrC1f),
                                       ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToGridf_scalar.restype = None

    xyzs, xyzt, area, rEs, iEs = extractorScalar(source, target, field, ctypes.c_float)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = arrC1f()

    lib.propagateToGridf_scalar(ctypes.byref(res),
                            xyzt[0], xyzt[1], xyzt[2],
                            xyzs[0], xyzs[1], xyzs[2],
                            rEs, iEs,
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    E = arrC1ToObj(res, shape=target_shape)

    return E

def calcFF_CPUf(source, target, currents, k, numThreads, epsilon, t_direction):
    lib = loadCPUlib()

    target_shape = (target.shape[0], target.shape[1])

    # Define arg and return types
    lib.propagateToFarFieldf.argtypes = [ctypes.POINTER(c2Bundlef),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                       ctypes.c_float, ctypes.c_float]

    lib.propagateToFarFieldf.restype = None

    xyzs, xyzt, area, rJs, iJs, rMs, iMs = extractor(source, target, currents, ctypes.c_float, normals=False)

    gs          = ctypes.c_int(source.shape[0] * source.shape[1])
    gt          = ctypes.c_int(target.shape[0] * target.shape[1])
    k           = ctypes.c_float(k)
    numThreads  = ctypes.c_int(numThreads)
    epsilon     = ctypes.c_float(epsilon)
    t_direction = ctypes.c_float(t_direction)

    # We pass reference to struct to c-function.
    res = c2Bundlef()

    lib.propagateToFarFieldf(ctypes.byref(res),
                            xyzt[0], xyzt[1],
                            xyzs[0], xyzs[1], xyzs[2],
                            rJs[0], rJs[1], rJs[2],
                            iJs[0], iJs[1], iJs[2],
                            rMs[0], rMs[1], rMs[2],
                            iMs[0], iMs[1], iMs[2],
                            area, gt, gs, k, numThreads,
                            epsilon, t_direction)

    # Unpack filled struct
    EH = c2BundleToObj(res, shape=target_shape, obj_t='fields')

    return EH

if __name__ == "__main__":
    rint("Bindings for POPPy CPU.")
