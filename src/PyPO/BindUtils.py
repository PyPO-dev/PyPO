import ctypes
import numpy as np

from src.PyPO.PyPOTypes import *

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

def fieldConv(field, c_fields, size, ct_t):
    c_fields.r1x = (ct_t * size)(*np.real(field.Ex).ravel().tolist())
    c_fields.r1y = (ct_t * size)(*np.real(field.Ey).ravel().tolist())
    c_fields.r1z = (ct_t * size)(*np.real(field.Ez).ravel().tolist())

    c_fields.i1x = (ct_t * size)(*np.imag(field.Ex).ravel().tolist())
    c_fields.i1y = (ct_t * size)(*np.imag(field.Ey).ravel().tolist())
    c_fields.i1z = (ct_t * size)(*np.imag(field.Ez).ravel().tolist())

    c_fields.r2x = (ct_t * size)(*np.real(field.Hx).ravel().tolist())
    c_fields.r2y = (ct_t * size)(*np.real(field.Hy).ravel().tolist())
    c_fields.r2z = (ct_t * size)(*np.real(field.Hz).ravel().tolist())

    c_fields.i2x = (ct_t * size)(*np.imag(field.Hx).ravel().tolist())
    c_fields.i2y = (ct_t * size)(*np.imag(field.Hy).ravel().tolist())
    c_fields.i2z = (ct_t * size)(*np.imag(field.Hz).ravel().tolist())

def sfieldConv(field, c_field, size, ct_t):
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

def allocate_reflcontainer(res, size, ct_t):
    res.size = size

    res.x = (ct_t * size)()
    res.y = (ct_t * size)()
    res.z = (ct_t * size)()

    res.nx = (ct_t * size)()
    res.ny = (ct_t * size)()
    res.nz = (ct_t * size)()

    res.area = (ct_t * size)()

def allocate_cframe(res, size, ct_t):
    res.size = size

    res.x = (ct_t * size)()
    res.y = (ct_t * size)()
    res.z = (ct_t * size)()

    res.dx = (ct_t * size)()
    res.dy = (ct_t * size)()
    res.dz = (ct_t * size)()

def allfill_cframe(res, frame_py, size, ct_t):
    res.size = size

    res.x = (ct_t * size)(*(frame_py.x.tolist()))
    res.y = (ct_t * size)(*(frame_py.y.tolist()))
    res.z = (ct_t * size)(*(frame_py.z.tolist()))

    res.dx = (ct_t * size)(*(frame_py.dx.tolist()))
    res.dy = (ct_t * size)(*(frame_py.dy.tolist()))
    res.dz = (ct_t * size)(*(frame_py.dz.tolist()))

def allfill_RTDict(res, rdict_py, ct_t):
    res.nRays = ctypes.c_int(rdict_py["nRays"])
    res.nRing = ctypes.c_int(rdict_py["nRing"])

    res.angx = ct_t(np.radians(rdict_py["angx"]))
    res.angy = ct_t(np.radians(rdict_py["angy"]))
    res.a = ct_t(rdict_py["a"])
    res.b = ct_t(rdict_py["b"])

    res.tChief = (ct_t * 3)(*np.radians(rdict_py["tChief"]).tolist())
    res.oChief = (ct_t * 3)(*rdict_py["oChief"].tolist())

def allfill_GRTDict(res, grdict_py, ct_t):
    res.nRays = ctypes.c_int(rdict_py["nRays"])

    res.angx = ct_t(np.radians(rdict_py["angx"]))
    res.angy = ct_t(np.radians(rdict_py["angy"]))
    res.a = ct_t(rdict_py["a"])
    res.b = ct_t(rdict_py["b"])

    res.tChief = (ct_t * 3)(*np.radians(rdict_py["tChief"]).tolist())
    res.oChief = (ct_t * 3)(*rdict_py["oChief"].tolist())

def allfill_GDict(res, gdict_py, ct_t):
    res.lam = ct_t(gdict_py["lam"])
    res.w0x = ct_t(gdict_py["w0x"])
    res.w0y = ct_t(gdict_py["w0y"])
    res.n = ct_t(gdict_py["n"])
    res.E0 = ct_t(gdict_py["E0"])
    res.z = ct_t(gdict_py["z"])
    res.pol = (ct_t * 3)(*gdict_py["pol"].tolist())

def creflToObj(res, shape, np_t):

    x = np.ctypeslib.as_array(res.x, shape=shape).astype(np_t)
    y = np.ctypeslib.as_array(res.y, shape=shape).astype(np_t)
    z = np.ctypeslib.as_array(res.z, shape=shape).astype(np_t)

    nx = np.ctypeslib.as_array(res.nx, shape=shape).astype(np_t)
    ny = np.ctypeslib.as_array(res.ny, shape=shape).astype(np_t)
    nz = np.ctypeslib.as_array(res.nz, shape=shape).astype(np_t)

    area = np.ctypeslib.as_array(res.area, shape=shape).astype(np_t)

    return reflGrids(x, y, z, nx, ny, nz, area)

def frameToObj(res, np_t, shape):
    x = np.ctypeslib.as_array(res.x, shape=shape).astype(np_t)
    y = np.ctypeslib.as_array(res.y, shape=shape).astype(np_t)
    z = np.ctypeslib.as_array(res.z, shape=shape).astype(np_t)

    dx = np.ctypeslib.as_array(res.dx, shape=shape).astype(np_t)
    dy = np.ctypeslib.as_array(res.dy, shape=shape).astype(np_t)
    dz = np.ctypeslib.as_array(res.dz, shape=shape).astype(np_t)

    return frame(shape[0], x, y, z, dx, dy, dz)

class WaitSymbol(object):
    def __init__(self):
        self.symList = ["|", "/", "-", "\\"]
        self.period = len(self.symList)
        self.n = 0

    def getSymbol(self):
        out = self.symList[self.n]
        self.n += 1

        if self.n == self.period:
            self.n = 0

        return out


