import ctypes
import numpy as np

# Classes to represent used return structtypes
# DOUBLES
class arrC1(ctypes.Structure):
    _fields_ = [("rx", ctypes.POINTER(ctypes.c_double)),
                ("ry", ctypes.POINTER(ctypes.c_double))]

class arrR3(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("z", ctypes.POINTER(ctypes.c_double))]

class c2Bundle(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_double)),
                ("r1y", ctypes.POINTER(ctypes.c_double)),
                ("r1z", ctypes.POINTER(ctypes.c_double)),
                ("i1x", ctypes.POINTER(ctypes.c_double)),
                ("i1y", ctypes.POINTER(ctypes.c_double)),
                ("i1z", ctypes.POINTER(ctypes.c_double)),
                ("r2x", ctypes.POINTER(ctypes.c_double)),
                ("r2y", ctypes.POINTER(ctypes.c_double)),
                ("r2z", ctypes.POINTER(ctypes.c_double)),
                ("i2x", ctypes.POINTER(ctypes.c_double)),
                ("i2y", ctypes.POINTER(ctypes.c_double)),
                ("i2z", ctypes.POINTER(ctypes.c_double))]


class c4Bundle(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_double)),
                ("r1y", ctypes.POINTER(ctypes.c_double)),
                ("r1z", ctypes.POINTER(ctypes.c_double)),
                ("i1x", ctypes.POINTER(ctypes.c_double)),
                ("i1y", ctypes.POINTER(ctypes.c_double)),
                ("i1z", ctypes.POINTER(ctypes.c_double)),
                ("r2x", ctypes.POINTER(ctypes.c_double)),
                ("r2y", ctypes.POINTER(ctypes.c_double)),
                ("r2z", ctypes.POINTER(ctypes.c_double)),
                ("i2x", ctypes.POINTER(ctypes.c_double)),
                ("i2y", ctypes.POINTER(ctypes.c_double)),
                ("i2z", ctypes.POINTER(ctypes.c_double)),
                ("r3x", ctypes.POINTER(ctypes.c_double)),
                ("r3y", ctypes.POINTER(ctypes.c_double)),
                ("r3z", ctypes.POINTER(ctypes.c_double)),
                ("i3x", ctypes.POINTER(ctypes.c_double)),
                ("i3y", ctypes.POINTER(ctypes.c_double)),
                ("i3z", ctypes.POINTER(ctypes.c_double)),
                ("r4x", ctypes.POINTER(ctypes.c_double)),
                ("r4y", ctypes.POINTER(ctypes.c_double)),
                ("r4z", ctypes.POINTER(ctypes.c_double)),
                ("i4x", ctypes.POINTER(ctypes.c_double)),
                ("i4y", ctypes.POINTER(ctypes.c_double)),
                ("i4z", ctypes.POINTER(ctypes.c_double))]

class c2rBundle(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_double)),
                ("r1y", ctypes.POINTER(ctypes.c_double)),
                ("r1z", ctypes.POINTER(ctypes.c_double)),
                ("i1x", ctypes.POINTER(ctypes.c_double)),
                ("i1y", ctypes.POINTER(ctypes.c_double)),
                ("i1z", ctypes.POINTER(ctypes.c_double)),
                ("r2x", ctypes.POINTER(ctypes.c_double)),
                ("r2y", ctypes.POINTER(ctypes.c_double)),
                ("r2z", ctypes.POINTER(ctypes.c_double)),
                ("i2x", ctypes.POINTER(ctypes.c_double)),
                ("i2y", ctypes.POINTER(ctypes.c_double)),
                ("i2z", ctypes.POINTER(ctypes.c_double)),
                ("r3x", ctypes.POINTER(ctypes.c_double)),
                ("r3y", ctypes.POINTER(ctypes.c_double)),
                ("r3z", ctypes.POINTER(ctypes.c_double))]

class reflparams(ctypes.Structure):
    _fields_ = [("coeffs", ctypes.POINTER(ctypes.c_double)),
                ("lxu", ctypes.POINTER(ctypes.c_double)),
                ("lyv", ctypes.POINTER(ctypes.c_double)),
                ("n_cells", ctypes.POINTER(ctypes.c_int)),
                ("flip", ctypes.c_bool),
                ("gmode", ctypes.c_bool),
                ("type", ctypes.c_int),
                ("transf", ctypes.POINTER(ctypes.c_double))]

class reflcontainer(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("z", ctypes.POINTER(ctypes.c_double)),
                ("nx", ctypes.POINTER(ctypes.c_double)),
                ("ny", ctypes.POINTER(ctypes.c_double)),
                ("nz", ctypes.POINTER(ctypes.c_double)),
                ("area", ctypes.POINTER(ctypes.c_double))]

# FLOATS
class arrC1f(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float))]

class arrR3f(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float)),
                ("z", ctypes.POINTER(ctypes.c_float))]

class c2Bundlef(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_float)),
                ("r1y", ctypes.POINTER(ctypes.c_float)),
                ("r1z", ctypes.POINTER(ctypes.c_float)),
                ("i1x", ctypes.POINTER(ctypes.c_float)),
                ("i1y", ctypes.POINTER(ctypes.c_float)),
                ("i1z", ctypes.POINTER(ctypes.c_float)),
                ("r2x", ctypes.POINTER(ctypes.c_float)),
                ("r2y", ctypes.POINTER(ctypes.c_float)),
                ("r2z", ctypes.POINTER(ctypes.c_float)),
                ("i2x", ctypes.POINTER(ctypes.c_float)),
                ("i2y", ctypes.POINTER(ctypes.c_float)),
                ("i2z", ctypes.POINTER(ctypes.c_float))]


class c4Bundlef(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_float)),
                ("r1y", ctypes.POINTER(ctypes.c_float)),
                ("r1z", ctypes.POINTER(ctypes.c_float)),
                ("i1x", ctypes.POINTER(ctypes.c_float)),
                ("i1y", ctypes.POINTER(ctypes.c_float)),
                ("i1z", ctypes.POINTER(ctypes.c_float)),
                ("r2x", ctypes.POINTER(ctypes.c_float)),
                ("r2y", ctypes.POINTER(ctypes.c_float)),
                ("r2z", ctypes.POINTER(ctypes.c_float)),
                ("i2x", ctypes.POINTER(ctypes.c_float)),
                ("i2y", ctypes.POINTER(ctypes.c_float)),
                ("i2z", ctypes.POINTER(ctypes.c_float)),
                ("r3x", ctypes.POINTER(ctypes.c_float)),
                ("r3y", ctypes.POINTER(ctypes.c_float)),
                ("r3z", ctypes.POINTER(ctypes.c_float)),
                ("i3x", ctypes.POINTER(ctypes.c_float)),
                ("i3y", ctypes.POINTER(ctypes.c_float)),
                ("i3z", ctypes.POINTER(ctypes.c_float)),
                ("r4x", ctypes.POINTER(ctypes.c_float)),
                ("r4y", ctypes.POINTER(ctypes.c_float)),
                ("r4z", ctypes.POINTER(ctypes.c_float)),
                ("i4x", ctypes.POINTER(ctypes.c_float)),
                ("i4y", ctypes.POINTER(ctypes.c_float)),
                ("i4z", ctypes.POINTER(ctypes.c_float))]

class c2rBundlef(ctypes.Structure):
    _fields_ = [("r1x", ctypes.POINTER(ctypes.c_float)),
                ("r1y", ctypes.POINTER(ctypes.c_float)),
                ("r1z", ctypes.POINTER(ctypes.c_float)),
                ("i1x", ctypes.POINTER(ctypes.c_float)),
                ("i1y", ctypes.POINTER(ctypes.c_float)),
                ("i1z", ctypes.POINTER(ctypes.c_float)),
                ("r2x", ctypes.POINTER(ctypes.c_float)),
                ("r2y", ctypes.POINTER(ctypes.c_float)),
                ("r2z", ctypes.POINTER(ctypes.c_float)),
                ("i2x", ctypes.POINTER(ctypes.c_float)),
                ("i2y", ctypes.POINTER(ctypes.c_float)),
                ("i2z", ctypes.POINTER(ctypes.c_float)),
                ("r3x", ctypes.POINTER(ctypes.c_float)),
                ("r3y", ctypes.POINTER(ctypes.c_float)),
                ("r3z", ctypes.POINTER(ctypes.c_float))]

class reflparamsf(ctypes.Structure):
    _fields_ = [("coeffs", ctypes.POINTER(ctypes.c_float)),
                ("lxu", ctypes.POINTER(ctypes.c_float)),
                ("lyv", ctypes.POINTER(ctypes.c_float)),
                ("n_cells", ctypes.POINTER(ctypes.c_int)),
                ("flip", ctypes.c_bool),
                ("gmode", ctypes.c_bool),
                ("type", ctypes.c_int),
                ("transf", ctypes.POINTER(ctypes.c_float))]

class reflcontainerf(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float)),
                ("z", ctypes.POINTER(ctypes.c_float)),
                ("nx", ctypes.POINTER(ctypes.c_float)),
                ("ny", ctypes.POINTER(ctypes.c_float)),
                ("nz", ctypes.POINTER(ctypes.c_float)),
                ("area", ctypes.POINTER(ctypes.c_float))]
