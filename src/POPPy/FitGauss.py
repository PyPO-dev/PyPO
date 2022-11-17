import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pt

from src.POPPy.POPPyTypes import *
from src.POPPy.BindRefl import *

def calcEstimates(x, y, area, field_norm_dB):
    M0 = np.sum(field_norm_dB)

    xm = np.sum(x * field_norm_dB) / M0
    ym = np.sum(y * field_norm_dB) / M0

    Mxx = np.sum(x**2 * field_norm_dB - xm**2) / M0
    Myy = np.sum(y**2 * field_norm_dB - ym**2) / M0
    Mxy = np.sum(x*y * field_norm_dB - xm*ym) / M0

    D = 1 / (2 * (Mxx*Myy - Mxy**2))

    a = Myy * D 
    b = Mxx * D
    c = -Mxy * D

    Amp = 0 #M0 * np.sqrt(a * b - c**2) * area[0,0]
    theta = 0.5 * np.arctan(2 * c / (a - b))
    
    p = np.sqrt((a - b)**2 + 4 * c**2)

    x0 = np.sqrt(1 / (a + b - p))
    y0 = np.sqrt(1 / (a + b + p))

    return x0, y0, xm, ym, theta

def fitGaussAbs(field, surfaceObject, thres):
    global thres_g
    thres_g = thres

    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y
    area = grids.area

    field_norm = np.absolute(field) / np.max(np.absolute(field))

    # INIT GUESSES


    mask_f = 20*np.log10(field_norm) >= thres

    field_est = 20 * np.log10(field_norm) * mask_f
    x0, y0, xs, ys, theta = calcEstimates(x, y, area, field_est)


    #pt.imshow(20*np.log10(mask_f * field_norm))
    #pt.show()

    pars = np.array([x0, y0, xs, ys, theta])
    args = (field_norm, mask_f, x, y)

    out = opt.minimize(couplingAbs, x0=pars, args=args, method='L-BFGS-B', tol=1e-10)

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
    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
    b = -np.sin(2 * theta) / (4 * x0**2) + np.sin(2 * theta) / (4 * y0**2)
    c = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)
    Psi = np.exp(-(a*(x - xs)**2 + 2*b*(x - xs)*(y - ys) + c*(y - ys)**2))

    #Psi = np.exp(-(((x-xs)/x0*np.cos(theta) + (y-ys)/y0*np.sin(theta)))**2 -(((x-xs)/x0*np.sin(theta) + (y-ys)/y0*np.cos(theta)))**2) * mask

    #Psi = np.exp(-(((x-xs)/x0*np.cos(theta) + (y-ys)/y0*np.sin(theta)))**2 -(((x-xs)/x0*np.sin(theta) + (y-ys)/y0*np.cos(theta)))**2)
    #Psi = np.exp(-((x-xs)/x0)**2 -((y-ys)/y0)**2)
    return Psi

def generateGauss(fgs_out, surfaceObject):
    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y

    x0, y0, xs, ys, theta = fgs_out

    Psi = GaussAbs(x, y, x0, y0, xs, ys, theta)
    return Psi
