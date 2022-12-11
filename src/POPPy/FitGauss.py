import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pt

from functools import partial

from src.POPPy.POPPyTypes import *
from src.POPPy.BindRefl import *

def calcEstimates(x, y, area, field_norm):

    M0 = np.sum(field_norm)

    xm = np.sum(x * field_norm) / M0
    ym = np.sum(y * field_norm) / M0

    Mxx = np.sum(x**2 * field_norm - xm**2) / M0
    Myy = np.sum(y**2 * field_norm - ym**2) / M0
    Mxy = np.sum(x*y * field_norm - xm*ym) / M0

    D = 1 / (2 * (Mxx*Myy - Mxy**2))

    a = Myy * D 
    b = Mxx * D
    c = -Mxy * D

    Amp = 1 #M0 * np.sqrt(a * b - c**2) * area[0,0]
    theta = 0.5 * np.arctan(2 * c / (a - b))

    if a-b > 0: # Not HW1 but the largest axis corresponds to theta.
        theta += np.pi/2
    if theta < 0:
        theta += np.pi

    p = np.sqrt((a - b)**2 + 4 * c**2)

    _x0 = 1 / (a + b - p)
    _y0 = 1 / (a + b - p) 

    if _x0 <= 0 and _y0 > 0:
        _x0 = _y0

    elif _y0 <= 0 and _x0 > 0:
        _y0 = _x0

    elif _y0 <= 0 and _x0 <= 0:
        _x0 *= -1
        _y0 *= -1

    x0 = np.sqrt(_x0)
    y0 = np.sqrt(_y0)

    return x0, y0, xm, ym, theta

def fitGaussAbs(field, surfaceObject, thres):
    global thres_g
    thres_g = thres
    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y
    
    area = grids.area

    mask_f = np.absolute(field) >= 10**(thres/20) * np.max(np.absolute(field))

    field /= np.max(np.absolute(field))

    field_est = field[mask_f]
    x0, y0, xs, ys, theta = calcEstimates(x[mask_f], y[mask_f], area[mask_f], field_est)
    print(f"Initial estimate from image moments:\nmu_x = {xs}, mu_y = {ys}\nTheta = {theta}")

    p0 = [x0, y0, xs, ys, theta]

    xy = (x[mask_f], y[mask_f]) # Independent vars
    
    bounds = ([np.finfo(float).eps, np.finfo(float).eps, -np.inf, -np.inf, theta-np.pi/2], 
            [np.inf, np.inf, np.inf, np.inf, theta+np.pi/2])
    couplingMasked = partial(GaussAbs, mask_f, "linear")

    popt, pcov = opt.curve_fit(couplingMasked, xy, field_est.ravel(), p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))

    print(f"Fitted shift and rotation:\nmu_x = {popt[-3]}, mu_y = {popt[-2]}\nTheta = {popt[-1]}")
    return popt, perr

def GaussAbs(mask, mode, xy, x0, y0, xs, ys, theta):
    x, y = xy
    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
    c = np.sin(2 * theta) / (4 * x0**2) - np.sin(2 * theta) / (4 * y0**2)
    b = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)

    if mode == "dB":
        Psi = 20*np.log10(np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)))

    elif mode == "linear":

        Psi = np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2))
    return (Psi).ravel()

def generateGauss(fgs_out, surfaceObject, mode, mask=None):
    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y

    if mask == None:
        mask = np.ones(x.shape)

    x0, y0, xs, ys, theta = fgs_out

    Psi = GaussAbs(mask, mode, (x, y), x0, y0, xs, ys, theta)
    return Psi.reshape(x.shape)
