import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pt

from src.POPPy.POPPyTypes import *
from src.POPPy.BindRefl import *

def fitGaussAbs(field, surfaceObject, thres):
    global thres_g
    thres_g = thres

    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y

    field_norm = np.absolute(field) / np.max(np.absolute(field))

    # INIT GUESSES

    x0 = 33/3600 # as
    y0 = 33/3600
    xs = x[np.unravel_index(field_norm.argmax(), field_norm.shape)]
    ys = y[np.unravel_index(field_norm.argmax(), field_norm.shape)]
    theta = 0



    mask_f = 20*np.log10(field_norm) > thres

    #pt.imshow(20*np.log10(mask_f * field_norm))
    #pt.show()

    pars = np.array([x0, y0, xs, ys, theta])
    args = (field_norm, mask_f, x, y)

    out = opt.minimize(couplingAbs, x0=pars, args=args, method='L-BFGS-B')

    return out["x"]

def couplingAbs(pars, *args):
    x0, y0, xs, ys, theta = pars
    field, mask, x, y = args
    field = np.absolute(field)

    Psi = np.absolute(GaussAbs(x, y, x0, y0, xs, ys, theta))
    Psi_max = np.max(Psi)

    mask_P = 20*np.log10(Psi / Psi_max) > thres_g

    #pt.imshow(mask_P)
    #pt.show()

    #pt.imshow(20*np.log10(mask_P * Psi / Psi_max))
    #pt.show()

    coupling = np.absolute(np.sum(Psi*mask * field*mask))**2 / (np.sum(np.absolute(field*mask)**2) * np.sum(np.absolute(Psi*mask)**2))
    #print(coupling)

    #eta = coupling**2
    #coupling = np.sum(np.absolute(Psi*mask_P))**2 / (np.sum(np.absolute(field)**2))
    eta = coupling
    eps = np.absolute(1 - eta)
    return eps

def GaussAbs(x, y, x0, y0, xs, ys, theta):
    Psi = np.exp(-(((x-xs)/x0*np.cos(theta) + (y-ys)/y0*np.sin(theta)))**2 -(((x-xs)/x0*np.sin(theta) + (y-ys)/y0*np.cos(theta)))**2)
    #Psi = np.exp(-((x-xs)/x0)**2 -((y-ys)/y0)**2)
    return Psi

def generateGauss(fgs_out, surfaceObject):
    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y

    x0, y0, xs, ys, theta = fgs_out

    Psi = GaussAbs(x, y, x0, y0, xs, ys, theta)
    return Psi
