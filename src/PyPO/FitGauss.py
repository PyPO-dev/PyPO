import numpy as np
from scipy.optimize import fmin

from PyPO.PyPOTypes import *
from PyPO.BindRefl import *
from PyPO.MatUtils import *

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

    Mxx = np.sum(x**2 * field_norm) / M0 - xm**2
    Myy = np.sum(y**2 * field_norm) / M0 - ym**2
    Mxy = np.sum(x*y * field_norm) / M0 - xm*ym

    D = 1 / (2 * (Mxx*Myy - Mxy**2))

    a = Myy * D 
    b = Mxx * D
    c = -Mxy * D

    Amp = 1 #M0 * np.sqrt(a * b - c**2) * area[0,0]
    theta = 0.5 * np.arctan(2 * c / (a - b))

    #if a-b > 0: # Not HW1 but the largest axis corresponds to theta.
    #    theta += np.pi/2
    #if theta < 0:
    #    theta += np.pi
    
    p = np.sqrt((a - b)**2 + 4 * c**2)

    _x0 = 1 / (a + b - p)
    _y0 = 1 / (a + b + p) 

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
# @param ratio Allowed maximal ratio of fit to actual beam.
#
# @returns popt Optimal parameters for Gaussian.
# @returns perr Standard deviations of optimal parameters.
def fitGaussAbs(field, surfaceObject, thres, mode, ratio=1):
    global thres_g
    thres_g = thres
    
    # Normalize
    _field = np.absolute(field) / np.max(np.absolute(field))
    #grids = generateGrid(surfaceObject, transform=True, spheric=False)
    grids = generateGrid(surfaceObject, transform=True, spheric=False)

    x = grids.x
    y = grids.y
    
    area = grids.area

    if mode == "dB":
        fit_field = 20 * np.log10(_field)
        mask_f = fit_field >= thres 

    elif mode == "linear":
        fit_field = _field
        mask_f = fit_field >= 10**(thres/20)

    elif mode == "log":
        fit_field = np.log(_field)
        mask_f = fit_field >= np.log(10**(thres/20))

    idx_max = np.unravel_index(np.argmax(fit_field), fit_field.shape)

    x_max = x[idx_max]
    y_max = y[idx_max] 

    idx_rows, idx_cols = findConnectedSubsets(mask_f, 1, idx_max)
   
    _xmin = x[np.min(idx_cols), idx_max[0]]
    _xmax = x[np.max(idx_cols), idx_max[0]]
    
    _ymin = y[idx_max[1], np.min(idx_rows)]
    _ymax = y[idx_max[1], np.max(idx_rows)]
    
    x_cond = (x > _xmin) & (x < _xmax)
    y_cond = (y > _ymin) & (y < _ymax)
    _mask_f = mask_f & x_cond & y_cond
    
    xmask = x[mask_f] - x_max
    ymask = y[mask_f] - y_max
    
    x0, y0, xs, ys, theta = calcEstimates(x[_mask_f], y[_mask_f], area[_mask_f], fit_field[_mask_f])

    p0 = [x0, y0, xs, ys, theta, np.max(fit_field)]


    _ratio = ratio
    num = 0

    if ratio is None:
        args = (surfaceObject, fit_field, _mask_f, mode)
        popt = fmin(GaussAbs, p0, args, disp=False)

    else:
        while _ratio >= ratio:
            if mode == "dB":
                fit_field = 20 * np.log10(_field)
                mask_f = fit_field >= thres 

            elif mode == "linear":
                fit_field = _field
                mask_f = fit_field >= 10**(thres/20)

            elif mode == "log":
                fit_field = np.log(_field)
                mask_f = fit_field >= np.log(10**(thres/20))
            
            if num >= 1:
                _mask_f = mask_f & x_cond & y_cond
            else:
                _mask_f = mask_f

            args = (surfaceObject, fit_field, _mask_f, mode)
            popt = fmin(GaussAbs, p0, args, disp=False)

            _Psi = generateGauss(popt, surfaceObject, mode="linear")
            _ratio = np.sum(_Psi**2) / np.sum(_field**2)
            
            if num > 1:
                if thres < -3:
                    thres += 0.5
            
            num += 1
    
    perr = np.zeros(len(popt))#np.sqrt(np.diag(pcov))
    perr = np.append(perr, 0.0)

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
#def GaussAbs(mask, mode, xy, x0, y0, xs, ys, theta):
#    x, y = xy
#    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
#    c = np.sin(2 * theta) / (4 * x0**2) - np.sin(2 * theta) / (4 * y0**2)
#    b = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)
#
#    if mode == "dB":
#        Psi = 20*np.log10(np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)))
#
#    elif mode == "linear":
#        Psi = np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2))
#
#    elif mode == "log":
#        Psi = -(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)
#
#    return (Psi).ravel()

## 
# Generate Absolute Gaussian from parameters.
# Called in optimalisation.
#
# @param p0 Array containing parameters for Gaussian.
# @param args Extra arguments for defining Gaussian and fit.
#
# @returns epsilon Coupling of field with Gaussian.
def GaussAbs(p0, *args):
    surfaceObject, field_est, mask, mode = args
    Psi = generateGauss(p0, surfaceObject, mode)
    coup = np.sum(np.absolute(Psi[mask])**2) / np.sum(np.absolute(field_est[mask])**2)
    epsilon = np.absolute(1 - coup)
    
    num = np.sum(field_est[mask] * Psi[mask])**2
    normE = np.sum(np.absolute(field_est[mask])**2)
    normP = np.sum(np.absolute(Psi[mask])**2)

    c00 = num / normE / normP
    
    eta = np.absolute(c00)
    epsilon = np.absolute(1 - eta)
    
    return epsilon

##
# Generate a Gaussian from surface parameters.
#
# @param fgs_out Optimal Gaussian parameters.
# @param surfaceObject Surface on which Gaussian is defined.
# @param mode Whether to fit Gaussian in linear, decibel or logarithmic space.
# @param mask Mask to apply to Gaussian distribution.
#
# @returns Psi Gaussian distribution.
def generateGauss(p0, surfaceObject, mode):
    x0, y0, xs, ys, theta, amp = p0
    
    grids = generateGrid(surfaceObject, transform=True, spheric=False)
    x = grids.x
    y = grids.y
    
    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
    c = np.sin(2 * theta) / (4 * x0**2) - np.sin(2 * theta) / (4 * y0**2)
    b = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)

    if mode == "dB":
        Psi = 20*np.log10(np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)))

    elif mode == "linear":
        Psi = np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2))

    elif mode == "log":
        Psi = -(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)
    
    return Psi.reshape(x.shape)
