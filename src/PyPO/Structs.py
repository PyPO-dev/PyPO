import ctypes
import numpy as np

## 
# @file
# Definitions of data structures used in the ctypes interface.
# The structures come in double format for CPU and single format for GPU.
# These defintions are only for passing structures from Python to the C/C++ backend.

##
# Two arrays representing a 1D complex array of double.
class arrC1(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double))]

##
# Three arrays representing a 3D real array of double.
class arrR3(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("z", ctypes.POINTER(ctypes.c_double))]

##
# Twelve arrays representing two 3D complex arrays of double.
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

##
# Twenty-four arrays representing four 3D complex arrays of double.
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

##
# Fifteen arrays representing two 3D complex arrays and one 3D real array of double.
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

##
# Reflectorparameters used by C++ backend, double precision.
class reflparams(ctypes.Structure):
    _fields_ = [("coeffs", ctypes.POINTER(ctypes.c_double)),
                ("lxu", ctypes.POINTER(ctypes.c_double)),
                ("lyv", ctypes.POINTER(ctypes.c_double)),
                ("n_cells", ctypes.POINTER(ctypes.c_int)),
                ("flip", ctypes.c_bool),
                ("gmode", ctypes.c_int),
                ("gcenter", ctypes.POINTER(ctypes.c_double)),
                ("ecc_uv", ctypes.c_double),
                ("rot_uv", ctypes.c_double),
                ("type", ctypes.c_int),
                ("transf", ctypes.POINTER(ctypes.c_double))]

##
# Container for realised reflector grids and normals in double precision.
class reflcontainer(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("z", ctypes.POINTER(ctypes.c_double)),
                ("nx", ctypes.POINTER(ctypes.c_double)),
                ("ny", ctypes.POINTER(ctypes.c_double)),
                ("nz", ctypes.POINTER(ctypes.c_double)),
                ("area", ctypes.POINTER(ctypes.c_double))]

##
# Container for storing ray-trace frames.
class cframe(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_double)),
                ("y", ctypes.POINTER(ctypes.c_double)),
                ("z", ctypes.POINTER(ctypes.c_double)),
                ("dx", ctypes.POINTER(ctypes.c_double)),
                ("dy", ctypes.POINTER(ctypes.c_double)),
                ("dz", ctypes.POINTER(ctypes.c_double))]

##
# Parameters for initializing a tubular ray-trace frame.
class RTDict(ctypes.Structure):
    _fields_ = [("nRays", ctypes.c_int),
                ("nRing", ctypes.c_int),
                ("angx0", ctypes.c_double),
                ("angy0", ctypes.c_double),
                ("x0", ctypes.c_double),
                ("y0", ctypes.c_double)]

##
# Parameters for initializing a Gaussian ray-trace frame.
class GRTDict(ctypes.Structure):
    _fields_ = [("nRays", ctypes.c_int),
                ("angx0", ctypes.c_double),
                ("angy0", ctypes.c_double),
                ("x0", ctypes.c_double),
                ("y0", ctypes.c_double),
                ("seed", ctypes.c_int)]

##
# Parameters for initializing a Gaussian PO beam.
class GPODict(ctypes.Structure):
    _fields_ = [("lam", ctypes.c_double),
                ("w0x", ctypes.c_double),
                ("w0y", ctypes.c_double),
                ("n", ctypes.c_double),
                ("E0", ctypes.c_double),
                ("dxyz", ctypes.c_double),
                ("pol", ctypes.POINTER(ctypes.c_double))]

##
# Parameters for initializing a scalar Gaussian PO beam.
class ScalarGPODict(ctypes.Structure):
    _fields_ = [("lam", ctypes.c_double),
                ("w0x", ctypes.c_double),
                ("w0y", ctypes.c_double),
                ("n", ctypes.c_double),
                ("E0", ctypes.c_double),
                ("dxyz", ctypes.c_double)]

##
# Two arrays representing a 1D complex array of floats.
class arrC1f(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float))]

##
# Three arrays representing a 3D array of floats.
class arrR3f(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float)),
                ("z", ctypes.POINTER(ctypes.c_float))]

##
# Twelve arrays representing two 3D complex arrays of float.
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


##
# Twenty-four arrays representing four 3D complex arrays of float.
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

##
# Fifteen arrays representing two 3D complex arrays and one 3D real array of float.
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

##
# Reflectorparameters used by C++ backend, single precision.
class reflparamsf(ctypes.Structure):
    _fields_ = [("coeffs", ctypes.POINTER(ctypes.c_float)),
                ("lxu", ctypes.POINTER(ctypes.c_float)),
                ("lyv", ctypes.POINTER(ctypes.c_float)),
                ("n_cells", ctypes.POINTER(ctypes.c_int)),
                ("flip", ctypes.c_bool),
                ("gmode", ctypes.c_int),
                ("gcenter", ctypes.POINTER(ctypes.c_float)),
                ("ecc_uv", ctypes.c_float),
                ("rot_uv", ctypes.c_float),
                ("type", ctypes.c_int),
                ("transf", ctypes.POINTER(ctypes.c_float))]

##
# Container for realised reflector grids and normals in single precision.
class reflcontainerf(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float)),
                ("z", ctypes.POINTER(ctypes.c_float)),
                ("nx", ctypes.POINTER(ctypes.c_float)),
                ("ny", ctypes.POINTER(ctypes.c_float)),
                ("nz", ctypes.POINTER(ctypes.c_float)),
                ("area", ctypes.POINTER(ctypes.c_float))]

##
# Container for storing ray-trace frames.
class cframef(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("x", ctypes.POINTER(ctypes.c_float)),
                ("y", ctypes.POINTER(ctypes.c_float)),
                ("z", ctypes.POINTER(ctypes.c_float)),
                ("dx", ctypes.POINTER(ctypes.c_float)),
                ("dy", ctypes.POINTER(ctypes.c_float)),
                ("dz", ctypes.POINTER(ctypes.c_float))]

##
# Parameters for initializing a tubular ray-trace frame.
class RTDictf(ctypes.Structure):
    _fields_ = [("nRays", ctypes.c_int),
                ("nRing", ctypes.c_int),
                ("angx0", ctypes.c_float),
                ("angy0", ctypes.c_float),
                ("x0", ctypes.c_float),
                ("y0", ctypes.c_float)]

##
# Parameters for initializing a Gaussian ray-trace frame.
class GRTDictf(ctypes.Structure):
    _fields_ = [("nRays", ctypes.c_int),
                ("angx0", ctypes.c_float),
                ("angy0", ctypes.c_float),
                ("x0", ctypes.c_float),
                ("y0", ctypes.c_float),
                ("seed", ctypes.c_int)]

##
# Parameters for initializing a Gaussian PO beam.
class GPODictf(ctypes.Structure):
    _fields_ = [("lam", ctypes.c_float),
                ("w0x", ctypes.c_float),
                ("w0y", ctypes.c_float),
                ("n", ctypes.c_float),
                ("E0", ctypes.c_float),
                ("dxyz", ctypes.c_float),
                ("pol", ctypes.POINTER(ctypes.c_float))]

##
# Parameters for initializing a scalar Gaussian PO beam.
class ScalarGPODictf(ctypes.Structure):
    _fields_ = [("lam", ctypes.c_float),
                ("w0x", ctypes.c_float),
                ("w0y", ctypes.c_float),
                ("n", ctypes.c_float),
                ("E0", ctypes.c_float),
                ("dxyz", ctypes.c_float)]
