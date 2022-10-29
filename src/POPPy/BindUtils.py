import ctypes
import numpy as np

from src.POPPy.POPPyTypes import *
from src.POPPy.Copy import copyGrid

# UTILITY FUNCTIONS

def currentConv(currents, c_currents, size, ct_t):
    c_currents.r1x = (ct_t * size)(*np.real(currents.Jx).ravel().tolist())
    c_currents.r1y = (ct_t * size)(*np.real(currents.Jy).ravel().tolist())
    c_currents.r1z = (ct_t * size)(*np.real(currents.Jz).ravel().tolist())

    c_currents.i1x = (ct_t * size)(*np.imag(currents.Jx).ravel().tolist())
    c_currents.i1y = (ct_t * size)(*np.imag(currents.Jy).ravel().tolist())
    c_currents.i1z = (ct_t * size)(*np.imag(currents.Jz).ravel().tolist())

    c_currents.r2x = (ct_t * size)(*np.real(currents.Mx).ravel().tolist())
    c_currents.r2y = (ct_t * size)(*np.real(currents.My).ravel().tolist())
    c_currents.r2z = (ct_t * size)(*np.real(currents.Mz).ravel().tolist())

    c_currents.i2x = (ct_t * size)(*np.imag(currents.Mx).ravel().tolist())
    c_currents.i2y = (ct_t * size)(*np.imag(currents.My).ravel().tolist())
    c_currents.i2z = (ct_t * size)(*np.imag(currents.Mz).ravel().tolist())

def fieldConv(field, c_field, size, ct_t):
    c_currents.rx = (ct_t * size)(*np.real(field.x).ravel().tolist())
    c_currents.ix = (ct_t * size)(*np.imag(field.x).ravel().tolist())

def extractorScalar(source, target, field, ct_t):
    """
    (PUBLIC)
    Function to extract the source/target grids from system elements.
    Converts to ctypes double.

    @param  ->
        source              :   The system element from which to propagate.
        target              :   The system element to propagate to.
    """

    gs = source.shape[0] * source.shape[1]
    gt = target.shape[0] * target.shape[1]

    # First, extract source grids.

    xs = (ct_t * gs)(*(source.grid_x.ravel().tolist()))
    ys = (ct_t * gs)(*(source.grid_y.ravel().tolist()))
    zs = (ct_t * gs)(*(source.grid_z.ravel().tolist()))

    xyzs = [xs, ys, zs]

    area = (ct_t * gs)(*(source.area.ravel().tolist()))

    # Extract target grids and normals
    xt = (ct_t * gt)(*target.grid_x.ravel().tolist())
    yt = (ct_t * gt)(*target.grid_y.ravel().tolist())
    zt = (ct_t * gt)(*target.grid_z.ravel().tolist())

    xyzt = [xt, yt, zt]

    rEs = (ct_t * gs)(*np.real(field).ravel().tolist())
    iEs = (ct_t * gs)(*np.imag(field).ravel().tolist())

    return xyzs, xyzt, area, rEs, iEs

def arrC1ToObj(res, shape):
    x1 = np.ctypeslib.as_array(res.x, shape=shape) + 1j * np.ctypeslib.as_array(res.y, shape=shape)
    return x1

def c2BundleToObj(res, shape, obj_t):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape)

    if obj_t == 'currents':
        out = currents(x1, y1, z1, x2, y2, z2)

    elif obj_t == 'fields':
        out = fields(x1, y1, z1, x2, y2, z2)

    return out

def c4BundleToObj(res, shape):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape)

    x3 = np.ctypeslib.as_array(res.r3x, shape=shape) + 1j * np.ctypeslib.as_array(res.i3x, shape=shape)
    y3 = np.ctypeslib.as_array(res.r3y, shape=shape) + 1j * np.ctypeslib.as_array(res.i3y, shape=shape)
    z3 = np.ctypeslib.as_array(res.r3z, shape=shape) + 1j * np.ctypeslib.as_array(res.i3z, shape=shape)

    x4 = np.ctypeslib.as_array(res.r4x, shape=shape) + 1j * np.ctypeslib.as_array(res.i4x, shape=shape)
    y4 = np.ctypeslib.as_array(res.r4y, shape=shape) + 1j * np.ctypeslib.as_array(res.i4y, shape=shape)
    z4 = np.ctypeslib.as_array(res.r4z, shape=shape) + 1j * np.ctypeslib.as_array(res.i4z, shape=shape)

    out1 = currents(x1, y1, z1, x2, y2, z2)
    out2 = fields(x3, y3, z3, x4, y4, z4)

    return out1, out2

def c2rBundleToObj(res, shape):
    x1 = np.ctypeslib.as_array(res.r1x, shape=shape) + 1j * np.ctypeslib.as_array(res.i1x, shape=shape)
    y1 = np.ctypeslib.as_array(res.r1y, shape=shape) + 1j * np.ctypeslib.as_array(res.i1y, shape=shape)
    z1 = np.ctypeslib.as_array(res.r1z, shape=shape) + 1j * np.ctypeslib.as_array(res.i1z, shape=shape)

    x2 = np.ctypeslib.as_array(res.r2x, shape=shape) + 1j * np.ctypeslib.as_array(res.i2x, shape=shape)
    y2 = np.ctypeslib.as_array(res.r2y, shape=shape) + 1j * np.ctypeslib.as_array(res.i2y, shape=shape)
    z2 = np.ctypeslib.as_array(res.r2z, shape=shape) + 1j * np.ctypeslib.as_array(res.i2z, shape=shape)

    x3 = np.ctypeslib.as_array(res.r3x, shape=shape)
    y3 = np.ctypeslib.as_array(res.r3y, shape=shape)
    z3 = np.ctypeslib.as_array(res.r3z, shape=shape)

    out1 = fields(x1, y1, z1, x2, y2, z2)
    out2 = rfield(x3, y3, z3)

    return out1, out2

def allocate_arrC1(res, size, ct_t):
    res.rx = (ct_t * size)()
    res.ix = (ct_t * size)()

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

def allfill_reflparams(inp, reflparams_py, ct_t):
    inp.coeffs = (ct_t * 3)()
    inp.type = reflparams_py["type"]

    for i in range(3):
        inp.coeffs[i] = ct_t(reflparams_py["coeffs"][i])

    inp.lxu = (ct_t * 2)()
    inp.lyv = (ct_t * 2)()
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

        inp.n_cells[i] = ctypes.c_int(reflparams_py["gridsize"][i])

    inp.flip = ctypes.c_bool(reflparams_py["flip"])
    inp.gmode = ctypes.c_int(reflparams_py["gmode"])

    inp.transf = (ct_t * 16)()
    for i in range(16):
        inp.transf[i] = ct_t(reflparams_py["transf"].ravel()[i])

def allocate_reflcontainer(res, size, ct_t):
    res.size = size

    res.x = (ct_t * size)()
    res.y = (ct_t * size)()
    res.z = (ct_t * size)()

    res.nx = (ct_t * size)()
    res.ny = (ct_t * size)()
    res.nz = (ct_t * size)()

    res.area = (ct_t * size)()

def creflToObj(res, shape, np_t):

    x = np.ctypeslib.as_array(res.x, shape=shape).astype(np_t)
    y = np.ctypeslib.as_array(res.y, shape=shape).astype(np_t)
    z = np.ctypeslib.as_array(res.z, shape=shape).astype(np_t)

    nx = np.ctypeslib.as_array(res.nx, shape=shape).astype(np_t)
    ny = np.ctypeslib.as_array(res.ny, shape=shape).astype(np_t)
    nz = np.ctypeslib.as_array(res.nz, shape=shape).astype(np_t)

    area = np.ctypeslib.as_array(res.area, shape=shape).astype(np_t)

    return reflGrids(x, y, z, nx, ny, nz, area)
