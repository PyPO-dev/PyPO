import ctypes
import numpy as np

from PyPO.PyPOTypes import *
import PyPO.Config as Config

##
# @file
# Utilities for the ctypes interface. These methods are mostly for allocating and 
# deallocating data. Also, Python types and PyPO objects are converted to ctypes structs here and vice versa.
# After converting to a ctypes struct, the struct is passed to the C/C++ source code and converted to a proper C/C++ struct for further usage.

##
# Convert a PyPO scalarfield object to a ctypes struct. 
#
# @params field A PyPO scalarfield object.
# @params c_field A ctypes arrC1 or arrC1f struct.
# @param size Number of points in fields object.
# @param ct_t Type of the floating point numbers for ctypes.
#
# @see scalarfield
# @see arrC1
# @see arrC1f
def sfieldConv(field, c_field, size, ct_t):
    c_field.x = (ct_t * size)(*np.real(field.S).ravel().tolist())
    c_field.y = (ct_t * size)(*np.imag(field.S).ravel().tolist())

##
# Convert a ctypes arrC1 or arrC1f struct to a PyPO scalarfield.
# 
# @param res An arrC1 or arrC1f struct containing the scalarfield.
# @param shape The shape of the scalarfield.
# @param np_t Type of data in numpy array to be filled.
#
# @returns res PyPO scalarfield object.
#
# @see arrC1
# @see arrC1f
# @see scalarfield
def arrC1ToObj(res, shape, np_t):
    obj = np.ctypeslib.as_array(res.x, shape=shape) + 1j * np.ctypeslib.as_array(res.y, shape=shape)

    res = scalarfield(obj)

    return res

##
# Convert a ctypes c2Bundle or c2Bundlef to a PyPO fields or currents object.
#
# @param res A c2Bundle or c2Bundlef struct.
# @param shape Shape of the fields or currents object.
# @param obj_t Whether to convert to a fields or currents object.
# @param np_t Type of data in numpy array to be filled.
#
# @returns out A fields or currents object filled with incoming EH fields or JM currents..
#
# @see c2Bundle 
# @see c2Bundlef
# @see fields
# @see currents
def c2BundleToObj(res, shape, obj_t, np_t):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape).astype(np_t)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape).astype(np_t)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape).astype(np_t)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape).astype(np_t)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape).astype(np_t)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape).astype(np_t)

    if obj_t == 'currents':
        out = currents(x1, y1, z1, x2, y2, z2)

    elif obj_t == 'fields':
        out = fields(x1, y1, z1, x2, y2, z2)

    return out

##
# Convert a ctypes c4Bundle or c4Bundlef to a PyPO fields and currents object.
#
# @param res A c4Bundle or c4Bundlef struct.
# @param shape Shape of the fields and currents object.
# @param np_t Type of data in numpy array to be filled.
#
# @returns out1 A fields object filled with incoming EH fields.
# @returns out2 A currents object filled with JM currents.
#
# @see c4Bundle 
# @see c4Bundlef
# @see fields
# @see currents
def c4BundleToObj(res, shape, np_t):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape).astype(np_t)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape).astype(np_t)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape).astype(np_t)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape).astype(np_t)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape).astype(np_t)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape).astype(np_t)

    x3 = np.ctypeslib.as_array(res.r3x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i3x, shape=shape).astype(np_t)
    y3 = np.ctypeslib.as_array(res.r3y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i3y, shape=shape).astype(np_t)
    z3 = np.ctypeslib.as_array(res.r3z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i3z, shape=shape).astype(np_t)

    x4 = np.ctypeslib.as_array(res.r4x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i4x, shape=shape).astype(np_t)
    y4 = np.ctypeslib.as_array(res.r4y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i4y, shape=shape).astype(np_t)
    z4 = np.ctypeslib.as_array(res.r4z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i4z, shape=shape).astype(np_t)

    out1 = currents(x1, y1, z1, x2, y2, z2)
    out2 = fields(x3, y3, z3, x4, y4, z4)

    return out1, out2

##
# Convert a ctypes c2rBundle or c2rBundlef to a PyPO fields and rfield object.
# The rfield will be filled by the calculated Poynting vectors.
#
# @param res A c2Bundle or c2rBundlef struct.
# @param shape Shape of the fields and rfield object.
# @param np_t Type of data in numpy array to be filled.
#
# @returns out1 A fields object filled with reflected EH fields.
# @returns out2 An rfield object filled with reflected Poynting vectors.
#
# @see c2rBundle 
# @see c2rBundlef
# @see fields
# @see rfields
def c2rBundleToObj(res, shape, np_t):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape).astype(np_t)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape).astype(np_t)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape).astype(np_t)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape).astype(np_t)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape).astype(np_t)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape).astype(np_t) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape).astype(np_t)

    x3 = np.ctypeslib.as_array(res.r3x, shape=shape).astype(np_t)
    y3 = np.ctypeslib.as_array(res.r3y, shape=shape).astype(np_t)
    z3 = np.ctypeslib.as_array(res.r3z, shape=shape).astype(np_t)

    out1 = fields(x1, y1, z1, x2, y2, z2)
    out2 = rfield(x3, y3, z3)

    return out1, out2

##
# allocate a ctypes arrC1 or arrC1f struct. Struct is then passed to and filled by the C/C++ code.
#
# @param res The arrC1 or arrC1f struct.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see arrC1
# @see arrC1f
def allocate_arrC1(res, size, ct_t):
    res.x = (ct_t * size)()
    res.y = (ct_t * size)()

##
# allocate a ctypes c2Bundle or c2Bundlef struct. Struct is then passed to and filled by the C/C++ code.
#
# @param res The c2Bundle or c2Bundlef struct.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see c2Bundle
# @see c2Bundlef
def allocate_c2Bundle(res, size, ct_t):
    res.r1x = (ct_t * size)()
    res.r1y = (ct_t * size)()
    res.r1z = (ct_t * size)()

    res.i1x = (ct_t * size)()
    res.i1y = (ct_t * size)()
    res.i1z = (ct_t * size)()

    res.r2x = (ct_t * size)()
    res.r2y = (ct_t * size)()
    res.r2z = (ct_t * size)()

    res.i2x = (ct_t * size)()
    res.i2y = (ct_t * size)()
    res.i2z = (ct_t * size)()

##
# allocate a ctypes c4Bundle or c4Bundlef struct. Struct is then passed to and filled by the C/C++ code.
#
# @param res The c4Bundle or c4Bundlef struct.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see c4Bundle
# @see c4Bundlef
def allocate_c4Bundle(res, size, ct_t):
    res.r1x = (ct_t * size)()
    res.r1y = (ct_t * size)()
    res.r1z = (ct_t * size)()

    res.i1x = (ct_t * size)()
    res.i1y = (ct_t * size)()
    res.i1z = (ct_t * size)()

    res.r2x = (ct_t * size)()
    res.r2y = (ct_t * size)()
    res.r2z = (ct_t * size)()

    res.i2x = (ct_t * size)()
    res.i2y = (ct_t * size)()
    res.i2z = (ct_t * size)()

    res.r3x = (ct_t * size)()
    res.r3y = (ct_t * size)()
    res.r3z = (ct_t * size)()

    res.i3x = (ct_t * size)()
    res.i3y = (ct_t * size)()
    res.i3z = (ct_t * size)()

    res.r4x = (ct_t * size)()
    res.r4y = (ct_t * size)()
    res.r4z = (ct_t * size)()

    res.i4x = (ct_t * size)()
    res.i4y = (ct_t * size)()
    res.i4z = (ct_t * size)()

##
# allocate a ctypes c2rBundle or c2rBundlef struct. Struct is then passed to and filled by the C/C++ code.
#
# @param res The c2rBundle or c2rBundlef struct.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see c2rBundle
# @see c2rBundlef
def allocate_c2rBundle(res, size, ct_t):
    res.r1x = (ct_t * size)()
    res.r1y = (ct_t * size)()
    res.r1z = (ct_t * size)()

    res.i1x = (ct_t * size)()
    res.i1y = (ct_t * size)()
    res.i1z = (ct_t * size)()

    res.r2x = (ct_t * size)()
    res.r2y = (ct_t * size)()
    res.r2z = (ct_t * size)()

    res.i2x = (ct_t * size)()
    res.i2y = (ct_t * size)()
    res.i2z = (ct_t * size)()

    res.r3x = (ct_t * size)()
    res.r3y = (ct_t * size)()
    res.r3z = (ct_t * size)()

##
# Allocate and fill ctypes reflparams or reflparamsf from a reflDict.
#
# @param inp A ctypes reflparams or reflparamsf struct.
# @param reflparams_py PyPO reflDict dictionary.
# @param ct_t Type of point in struct.
#
# @see reflparams
# @see reflparamsf
# @see reflDict
def allfill_reflparams(inp, reflparams_py, ct_t):
    inp.coeffs = (ct_t * 3)()
    inp.type = ctypes.c_int(reflparams_py["type"])
    inp.ecc_uv = ct_t(reflparams_py["ecc_uv"])
    inp.rot_uv = ct_t(reflparams_py["rot_uv"])

    for i in range(3):
        inp.coeffs[i] = ct_t(reflparams_py["coeffs"][i])

    inp.lxu = (ct_t * 2)()
    inp.lyv = (ct_t * 2)()
    inp.gcenter = (ct_t * 2)()
    inp.n_cells = (ctypes.c_int * 2)()

    for i in range(2):
        if reflparams_py["gmode"] == 0:
            inp.lxu[i] = ct_t(reflparams_py["lims_x"][i])
            inp.lyv[i] = ct_t(reflparams_py["lims_y"][i])

        elif reflparams_py["gmode"] == 1:
            inp.lxu[i] = ct_t(reflparams_py["lims_u"][i])
            inp.lyv[i] = ct_t(reflparams_py["lims_v"][i])

        elif reflparams_py["gmode"] == 2:
            inp.lxu[i] = ct_t(reflparams_py["lims_Az"][i])
            inp.lyv[i] = ct_t(reflparams_py["lims_El"][i])

        inp.gcenter[i] = ct_t(reflparams_py["gcenter"][i])
        inp.n_cells[i] = ctypes.c_int(reflparams_py["gridsize"][i])

    inp.flip = ctypes.c_bool(reflparams_py["flip"])
    inp.gmode = ctypes.c_int(reflparams_py["gmode"])

    inp.transf = (ct_t * 16)()
    for i in range(16):
        inp.transf[i] = ct_t(reflparams_py["transf"].ravel()[i])

##
# Allocate ctypes reflcontainer or reflcontainerf.
#
# @param res A ctypes reflcontainer or reflcontainerf struct.
# @param size Number of points on reflector.
# @param ct_t Type of point in struct.
#
# @see reflcontainer
# @see reflcontainerf
# @see reflDict
def allocate_reflcontainer(res, size, ct_t):
    res.size = size

    res.x = (ct_t * size)()
    res.y = (ct_t * size)()
    res.z = (ct_t * size)()

    res.nx = (ct_t * size)()
    res.ny = (ct_t * size)()
    res.nz = (ct_t * size)()

    res.area = (ct_t * size)()

##
# Allocate a ctypes cframe or cframef struct.
#
# @param res A ctypes cframe or cframef struct.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see cframe
# @see cframef
def allocate_cframe(res, size, ct_t):
    res.size = size

    res.x = (ct_t * size)()
    res.y = (ct_t * size)()
    res.z = (ct_t * size)()

    res.dx = (ct_t * size)()
    res.dy = (ct_t * size)()
    res.dz = (ct_t * size)()

##
# Allocate and fill a ctypes cframe or cframef struct.
#
# @param res A ctypes cframe or cframef struct.
# @param frame_py A PyPO frame object.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see cframe
# @see cframef
# @see frame
def allfill_cframe(res, frame_py, size, ct_t):
    res.size = size

    res.x = (ct_t * size)(*(frame_py.x.tolist()))
    res.y = (ct_t * size)(*(frame_py.y.tolist()))
    res.z = (ct_t * size)(*(frame_py.z.tolist()))

    res.dx = (ct_t * size)(*(frame_py.dx.tolist()))
    res.dy = (ct_t * size)(*(frame_py.dy.tolist()))
    res.dz = (ct_t * size)(*(frame_py.dz.tolist()))

##
# Allocate and fill a ctypes c2Bundle or c2Bundleff struct.
#
# @param res A ctypes c2Bundle or c2Bundlef struct.
# @param obj_py a PyPO fields or currents object.
# @param size Number of points in struct.
# @param ct_t Type of point in struct.
#
# @see c2Bundle
# @see c2Bundlef
# @see fields
# @see currents
def allfill_c2Bundle(res, obj_py, size, ct_t):
    #*np.real(field.Ex).ravel().tolist()
    res.r1x = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[0])).ravel().tolist())
    res.r1y = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[1])).ravel().tolist())
    res.r1z = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[2])).ravel().tolist())
                                       
    res.i1x = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[0])).ravel().tolist())
    res.i1y = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[1])).ravel().tolist())
    res.i1z = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[2])).ravel().tolist())
    
    res.r2x = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[3])).ravel().tolist())
    res.r2y = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[4])).ravel().tolist())
    res.r2z = (ct_t * size)(*np.real(getattr(obj_py, obj_py.memlist[5])).ravel().tolist())
                                                                                
    res.i2x = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[3])).ravel().tolist())
    res.i2y = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[4])).ravel().tolist())
    res.i2z = (ct_t * size)(*np.imag(getattr(obj_py, obj_py.memlist[5])).ravel().tolist())

##
# Allocate and fill an RTDict struct, for generating a tubular ray-trace frame.
#
# @param res A RTDict or RTDictf struct.
# @param rdict_py A TubeRTDict.
# @param ct_t Type of point in struct.
#
# @see RTDict
# @see RTDictf
def allfill_RTDict(res, rdict_py, ct_t):
    res.nRays   = ctypes.c_int(rdict_py["nRays"])
    res.nRing   = ctypes.c_int(rdict_py["nRing"])

    res.angx0   = ct_t(np.radians(rdict_py["angx0"]))
    res.angy0   = ct_t(np.radians(rdict_py["angy0"]))
    res.x0      = ct_t(rdict_py["x0"])
    res.y0      = ct_t(rdict_py["y0"])

##
# Allocate and fill a GRTDict, for generating a Gaussian ray-trace frame.
#
# @param res A GRTDict or GRTDictf struct.
# @param grdict_py A GRTDict.
# @param ct_t Type of field in struct.
#
# @see GRTDict
# @see GRTDictf
def allfill_GRTDict(res, grdict_py, ct_t):
    res.nRays = ctypes.c_int(grdict_py["nRays"])

    res.angx0 = ct_t(np.radians(grdict_py["angx0"]))
    res.angy0 = ct_t(np.radians(grdict_py["angy0"]))
    res.x0 = ct_t(grdict_py["x0"])
    res.y0 = ct_t(grdict_py["y0"])
    res.seed = ctypes.c_int(grdict_py["seed"])

##
# Allocate and fill  a GPODict, for generating a Gaussian beam field and current.
#
# @param res A GPODict or GPODictf struct.
# @param gdict_py A GPODict.
# @param ct_t Type of field in struct.
#
# @see GPODict
# @see GPODictf
def allfill_GPODict(res, gdict_py, ct_t):
    res.lam = ct_t(gdict_py["lam"])
    res.w0x = ct_t(gdict_py["w0x"])
    res.w0y = ct_t(gdict_py["w0y"])
    res.n = ct_t(gdict_py["n"])
    res.E0 = ct_t(gdict_py["E0"])
    res.dxyz = ct_t(gdict_py["dxyz"])

    res.pol = (ct_t * 3)(*gdict_py["pol"].tolist())

##
# Allocate and fill  a ScalarGPODict, for generating a scalar Gaussian beam field.
#
# @param res A ScalarGPODict or ScalarGPODictf struct.
# @param sgdict_py A GPODict.
# @param ct_t Type of field in struct.
#
# @see ScalarGPODict
# @see ScalarGPODictf
def allfill_SGPODict(res, sgdict_py, ct_t):
    res.lam = ct_t(sgdict_py["lam"])
    res.w0x = ct_t(sgdict_py["w0x"])
    res.w0y = ct_t(sgdict_py["w0y"])
    res.n = ct_t(sgdict_py["n"])
    res.E0 = ct_t(sgdict_py["E0"])
    res.dxyz = ct_t(sgdict_py["dxyz"])

##
# Allocate and fill a 4D matrix for transforming frames and fields/currents.
#
# @param mat Matrix containing transformation.
# @param ct_t Type of field in matrix.
#
# @returns c_mat The ctypes representation of the matrix.
def allfill_mat4D(mat, ct_t):
    c_mat = (ct_t * 16)()
    
    for i in range(16):
        c_mat[i] = ct_t(mat.ravel()[i])

    return c_mat

##
# Convert a reflector grids struct to a PyPO grids object.
#
# @param res A reflcontainer or reflcontainerf struct.
# @param shape Shape of the reflector grid.
# @param np_t Type of field in PyPO object.
#
# @returns out The grids as PyPO object.
#
# @see reflcontainer
# @see reflcontainerf
# @see reflGrids 
def creflToObj(res, shape, np_t):

    x = np.ctypeslib.as_array(res.x, shape=shape).astype(np_t)
    y = np.ctypeslib.as_array(res.y, shape=shape).astype(np_t)
    z = np.ctypeslib.as_array(res.z, shape=shape).astype(np_t)

    nx = np.ctypeslib.as_array(res.nx, shape=shape).astype(np_t)
    ny = np.ctypeslib.as_array(res.ny, shape=shape).astype(np_t)
    nz = np.ctypeslib.as_array(res.nz, shape=shape).astype(np_t)

    area = np.ctypeslib.as_array(res.area, shape=shape).astype(np_t)
    out = reflGrids(x, y, z, nx, ny, nz, area)
    return out
##
# Convert a cframe struct to a PyPO frame object.
#
# @param res A cframe or cframef struct.
# @param np_t Type of field in PyPO frame object.
# @param shape Shape of resulting PyPO frame.
#
# @returns out PyPO frame object.
#
# @see cframe
# @see cframef
# @see frame
def frameToObj(res, np_t, shape):
    x = np.ctypeslib.as_array(res.x, shape=shape).astype(np_t)
    y = np.ctypeslib.as_array(res.y, shape=shape).astype(np_t)
    z = np.ctypeslib.as_array(res.z, shape=shape).astype(np_t)

    dx = np.ctypeslib.as_array(res.dx, shape=shape).astype(np_t)
    dy = np.ctypeslib.as_array(res.dy, shape=shape).astype(np_t)
    dz = np.ctypeslib.as_array(res.dz, shape=shape).astype(np_t)
    
    out = frame(shape[0], x, y, z, dx, dy, dz)

    return out 

