import ctypes
import math
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

#############################################################################
#                                                                           #
#              List of bindings for the GPU interface of PyPO.             #
#                                                                           #
#############################################################################

def loadGPUlib():
    try:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build/Debug"
        lib = ctypes.CDLL(str(LD_PATH/"pypogpu.dll"))
    except:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build"
        
        try:
            lib = ctypes.CDLL(LD_PATH/"libpypogpu.so")
        except:
            lib = ctypes.CDLL(LD_PATH/"libpypogpu.dylib")
    
    lib.callKernelf_JM.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JM.restype = None

    lib.callKernelf_EH.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EH.restype = None

    lib.callKernelf_JMEH.argtypes = [ctypes.POINTER(c4Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_JMEH.restype = None

    lib.callKernelf_EHP.argtypes = [ctypes.POINTER(c2rBundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_EHP.restype = None

    lib.callKernelf_FF.argtypes = [ctypes.POINTER(c2Bundlef), reflparamsf, reflparamsf,
                                   ctypes.POINTER(reflcontainerf), ctypes.POINTER(reflcontainerf),
                                   ctypes.POINTER(c2Bundlef), ctypes.c_float, ctypes.c_float,
                                   ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callKernelf_FF.restype = None

    lib.callRTKernel.argtypes = [reflparamsf, ctypes.POINTER(cframef), ctypes.POINTER(cframef),
                                ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int]

    lib.callRTKernel.restype = None

    ws = WaitSymbol()

    return lib, ws

# WRAPPER FUNCTIONS DOUBLE PREC

#### SINGLE PRECISION
def PyPO_GPUf(source, target, PODict):
    lib, ws = loadGPUlib()
    mgr = TManager.Manager(Config.context)

    # Create structs with pointers for source and target
    csp = reflparamsf()
    ctp = reflparamsf()

    cs = reflcontainerf()
    ct = reflcontainerf()

    gs = source["gridsize"][0] * source["gridsize"][1]
    gt = target["gridsize"][0] * target["gridsize"][1]

    allfill_reflparams(csp, source, ctypes.c_float)
    allfill_reflparams(ctp, target, ctypes.c_float)

    allocate_reflcontainer(cs, gs, ctypes.c_float)
    allocate_reflcontainer(ct, gt, ctypes.c_float)

    if PODict["mode"] == "Scalar":
        c_cfield = arrC1f()
        fieldConv(PODict["s_field"], c_field, gs, ctypes.c_float)

    else:
        c_currents = c2Bundlef()
        currentConv(PODict["s_current"], c_currents, gs, ctypes.c_float)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    nBlocks = math.ceil(gt / PODict["nThreads"])

    if PODict["exp"] == "fwd":
        exp_prop = -1

    elif PODict["exp"] == "bwd":
        exp_prop = 1

    k           = ctypes.c_float(PODict["k"])
    nThreads    = ctypes.c_int(PODict["nThreads"])
    nBlocks     = ctypes.c_int(nBlocks)
    epsilon     = ctypes.c_float(PODict["epsilon"])
    t_direction = ctypes.c_float(exp_prop)

    args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
            ctypes.byref(c_currents), k, epsilon,
            t_direction, nBlocks, nThreads]


    if PODict["mode"] == "JM":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        #t = mgr.new_sthread(target=lib.callKernelf_JM, args=args)
        mgr.new_sthread(target=lib.callKernelf_JM, args=args, calc_type="currents")
        
        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif PODict["mode"] == "EH":
        res = c2Bundlef()

        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_EH, args=args, calc_type="fields")
        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif PODict["mode"] == "JMEH":
        res = c4Bundlef()

        args.insert(0, res)

        allocate_c4Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_JMEH, args=args, calc_type="currents & fields")
        """
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating J, M, E, H on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated J, M, E, H on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')
        """
        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif PODict["mode"] == "EHP":
        res = c2rBundlef()

        args.insert(0, res)

        allocate_c2rBundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_EHP, args=args, calc_type="reflected fields & Poynting")
        #t = threading.Thread(target=lib.callKernelf_EHP, args=args)
        #t.daemon = True
        #t.start()
        #while t.is_alive(): # wait for the thread to exit
        #    Config.print(f'Calculating reflected E, H, P on {target["name"]} {ws.getSymbol()}', end='\r')
        #    t.join(.1)
        #dtime = time.time() - start_time
        #Config.print(f'Calculated reflected E, H, P on {target["name"]} in {dtime:.3f} seconds', end='\r')
        #Config.print(f'\n')

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif PODict["mode"] == "FF":
        res = c2Bundlef()
        args.insert(0, res)

        allocate_c2Bundle(res, target["gridsize"][0] * target["gridsize"][1], ctypes.c_float)

        mgr.new_sthread(target=lib.callKernelf_FF, args=args, calc_type="far-field")
        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

def RT_GPUf(target, fr_in, epsilon, t0, nThreads):
    lib, ws = loadGPUlib()

    ctp = reflparamsf()
    allfill_reflparams(ctp, target, ctypes.c_float)

    inp = cframef()
    res = cframef()

    allocate_cframe(res, fr_in.size, ctypes.c_float)
    allfill_cframe(inp, fr_in, fr_in.size, ctypes.c_float)

    nBlocks = math.ceil(fr_in.size / nThreads)

    nBocks      = ctypes.c_int(nBlocks)
    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_float(epsilon)
    t0          = ctypes.c_float(t0)

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
            epsilon, t0, nBlocks, nThreads]

    start_time = time.time()
    
    t = threading.Thread(target=lib.callRTKernel, args=args)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exitp
        Config.print(f'Calculating ray-trace to {target["name"]} {ws.getSymbol()}', end='\r')
        t.join(.1)
    dtime = time.time() - start_time
    Config.print(f'Calculated ray-trace to {target["name"]} in {dtime:.3f} seconds', end='\r')
    Config.print(f'\n')

    shape = (fr_in.size,)
    fr_out = frameToObj(res, np_t=np.float32, shape=shape)

    return fr_out

if __name__ == "__main__":
    print("Bindings for PyPO GPU.")
