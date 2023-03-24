import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as pt

from functools import partial

from src.PyPO.PyPOTypes import *
from src.PyPO.BindRefl import *

##
# @file
# File containing methods for fitting Gaussian distributions to field components.

##
# Calculate estimates for beam parameters from an input field component.
# These estimates are used as initial values for Gaussian fitting.
#
# @param x Grid of x co-ordinates of input field.
# @param y Grid of y co-ordinates of input field.
# @param area Grid of area elements.
# @param field_norm Normalised input field component.
#
# @returns x0 Semi-major axis size of estimate.
# @returns y0 Semi-minor axis size of estimate.
# @returns xm Mean x-center of estimate.
# @returns ym Mean y-center of estimate.
# @returns theta Position angle of estimate.
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

##
# Fit a Gaussian to an amplitude pattern of a field component.
#
# @param field Component of field to fit.
# @param surfaceObject Surface on which the field is defined.
# @param thres Threshold for fitting in decibels.
# @param mode Whether to fit the Gaussian in linear, decibel or logarithmic space.
#
# @returns popt Optimal parameters for Gaussian.
# @returns perr Standard deviations of optimal parameters.
def fitGaussAbs(field, surfaceObject, thres, mode):
    global thres_g
    thres_g = thres
    
    # Normalize
    field = np.absolute(field) / np.max(np.absolute(field))

    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y
    
    area = grids.area

    if mode == "dB":
        fit_field = 20 * np.log10(field)
        mask_f = fit_field >= thres 

    elif mode == "linear":
        fit_field = field
        mask_f = fit_field >= 10**(thres/20)

    elif mode == "log":
        fit_field = np.log(field)
        mask_f = fit_field >= np.log(10**(thres/20))

    field_est = fit_field[mask_f]
    x0, y0, xs, ys, theta = calcEstimates(x[mask_f], y[mask_f], area[mask_f], field_est)
    #print(f"Initial estimate from image moments:\nmu_x = {xs}, mu_y = {ys}\nTheta = {theta}")

    p0 = [x0, y0, xs, ys, theta]

    xy = (x[mask_f], y[mask_f]) # Independent vars
    
    bounds = ([np.finfo(float).eps, np.finfo(float).eps, -np.inf, -np.inf, theta-np.pi/4], 
            [np.inf, np.inf, np.inf, np.inf, theta+np.pi/4])
    couplingMasked = partial(GaussAbs, mask_f, mode)

    popt, pcov = opt.curve_fit(couplingMasked, xy, field_est.ravel(), p0, bounds=bounds, method="dogbox")
    perr = np.sqrt(np.diag(pcov))

    popt = np.append(popt, np.max(np.absolute(field)))
    perr = np.append(perr, 0.0)

    #print(f"Fitted shift and rotation:\nmu_x = {popt[-3]}, mu_y = {popt[-2]}\nTheta = {popt[-1]}")
    return popt, perr

##
# Generate a Gaussian pattern from Gaussian parameters.
#
# @param mask Mask to apply to generated Gaussian.
# @param mode Whether to generate the Gaussian in linear, decibel or logarithmic space.
# @param xy Tuple containing x and y grids of surface of Gaussian.
# @param x0 Gaussian beamwidth in x-direction.
# @param y0 Gaussian beamwidth in y-direction.
# @param xs Center in x-direction of Gaussian.
# @param ys Center in y-direction of Gaussian.
# @param theta Position angle of Gaussian.
#
# @returns Psi The Gaussian distribution.
def GaussAbs(mask, mode, xy, x0, y0, xs, ys, theta):
    x, y = xy
    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
    c = np.sin(2 * theta) / (4 * x0**2) - np.sin(2 * theta) / (4 * y0**2)
    b = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)

    if mode == "dB":
        Psi = 20*np.log10(np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)))

    elif mode == "linear":
        Psi = np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2))

    elif mode == "log":
        Psi = -(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)

    return (Psi).ravel()

##
# Generate a Gaussian from surface parameters.
#
# @param fgs_out Optimal Gaussian parameters.
# @param surfaceObject Surface on which Gaussian is defined.
# @param mode Whether to fit Gaussian in linear, decibel or logarithmic space.
# @param mask Mask to apply to Gaussian distribution.
#
# @returns Psi Gaussian distribution.
def generateGauss(fgs_out, surfaceObject, mode, mask=None):
    grids = generateGrid(surfaceObject, transform=False, spheric=False)

    x = grids.x
    y = grids.y

    if mask == None:
        mask = np.ones(x.shape)

    x0, y0, xs, ys, theta, amp = fgs_out

    Psi = GaussAbs(mask, mode, (x, y), x0, y0, xs, ys, theta)
    return amp * Psi.reshape(x.shape)
