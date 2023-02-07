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

import threading

#############################################################################
#                                                                           #
#              List of bindings for the CPU interface of PyPO.             #
#                                                                           #
#############################################################################

def loadCPUlib():
    try:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build/Debug"
        lib = ctypes.CDLL(str(LD_PATH/"pypocpu.dll"))
    except:
        LD_PATH = pathlib.Path(__file__).parents[2]/"out/build"
        try:
            lib = ctypes.CDLL(LD_PATH/"libpypocpu.so")
        except:
            lib = ctypes.CDLL(LD_PATH/"libpypocpu.dylib")

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

    ws = WaitSymbol()

    return lib, ws

# WRAPPER FUNCTIONS DOUBLE PREC
def PyPO_CPUd(source, target, PODict):
    lib, ws = loadCPUlib()

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

    if PODict["mode"] == "Scalar":
        c_cfield = arrC1()
        fieldConv(PODict["s_field"], c_field, gs, ctypes.c_double)

    else:
        c_currents = c2Bundle()
        currentConv(PODict["s_current"], c_currents, gs, ctypes.c_double)

    target_shape = (target["gridsize"][0], target["gridsize"][1])

    if PODict["exp"] == "fwd":
        exp_prop = -1

    elif PODict["exp"] == "bwd":
        exp_prop = 1

    k           = ctypes.c_double(PODict["k"])
    nThreads    = ctypes.c_int(PODict["nThreads"])
    epsilon     = ctypes.c_double(PODict["epsilon"])
    t_direction = ctypes.c_double(exp_prop)

    args = [csp, ctp, ctypes.byref(cs), ctypes.byref(ct),
            ctypes.byref(c_currents), k, nThreads, epsilon,
            t_direction]
    start_time = time.time()

    if PODict["mode"] == "JM":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_JM, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating J, M on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated J, M on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        JM = c2BundleToObj(res, shape=target_shape, obj_t='currents', np_t=np.float64)

        return JM

    elif PODict["mode"] == "EH":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_EH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating E, H on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated E, H on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

    elif PODict["mode"] == "JMEH":
        res = c4Bundle()
        args.insert(0, res)

        allocate_c4Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_JMEH, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating J, M, E, H on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated J, M, E, H on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        JM, EH = c4BundleToObj(res, shape=target_shape, np_t=np.float64)

        return [JM, EH]

    elif PODict["mode"] == "EHP":
        res = c2rBundle()
        args.insert(0, res)

        allocate_c2rBundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_EHP, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating reflected E, H, P on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated reflected E, H, P on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        EH, Pr = c2rBundleToObj(res, shape=target_shape, np_t=np.float64)

        return [EH, Pr]

    elif PODict["mode"] == "Scalar":
        res = arrC1()
        args.insert(0, res)

        allocate_arrC1(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToGrid_scalar, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating scalar field on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated scalar field on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        E = arrC1ToObj(res, shape=target_shape)

        return E

    elif PODict["mode"] == "FF":
        res = c2Bundle()
        args.insert(0, res)

        allocate_c2Bundle(res, gt, ctypes.c_double)

        t = threading.Thread(target=lib.propagateToFarField, args=args)
        t.daemon = True
        t.start()
        while t.is_alive(): # wait for the thread to exit
            Config.print(f'Calculating far-field E, H on {target["name"]} {ws.getSymbol()}', end='\r')
            t.join(.1)
        dtime = time.time() - start_time
        Config.print(f'Calculated far-field E, H on {target["name"]} in {dtime:.3f} seconds', end='\r')
        Config.print(f'\n')

        # Unpack filled struct
        EH = c2BundleToObj(res, shape=target_shape, obj_t='fields', np_t=np.float64)

        return EH

def RT_CPUd(target, fr_in, epsilon, t0, nThreads):
    lib, ws = loadCPUlib()

    inp = cframe()
    res = cframe()

    allocate_cframe(res, fr_in.size, ctypes.c_double)
    allfill_cframe(inp, fr_in, fr_in.size, ctypes.c_double)

    ctp = reflparams()
    allfill_reflparams(ctp, target, ctypes.c_double)

    nThreads    = ctypes.c_int(nThreads)
    epsilon     = ctypes.c_double(epsilon)
    t0          = ctypes.c_double(t0)

    args = [ctp, ctypes.byref(inp), ctypes.byref(res),
                        nThreads, epsilon, t0]
    start_time = time.time()
    
    t = threading.Thread(target=lib.propagateRays, args=args)
    t.daemon = True
    t.start()
    while t.is_alive(): # wait for the thread to exit
        Config.print(f'Calculating ray-trace to {target["name"]} {ws.getSymbol()}', end='\r')
        t.join(.1)
    dtime = time.time() - start_time
    Config.print(f'Calculated ray-trace to {target["name"]} in {dtime:.3f} seconds', end='\r')
    Config.print(f'\n')

    shape = (fr_in.size,)
    fr_out = frameToObj(res, np_t=np.float64, shape=shape)
    return fr_out

if __name__ == "__main__":
    print("Bindings for PyPO CPU.")
