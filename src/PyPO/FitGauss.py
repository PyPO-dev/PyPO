"""!
@file
File containing methods for fitting Gaussian distributions to field components.
"""

import numpy as np
from copy import deepcopy
from scipy.optimize import fmin
from scipy.interpolate import griddata

import PyPO.BindRefl as BRefl
import PyPO.MatUtils as MUtils
from PyPO.Enums import Scales

def calcEstimates(x, y, field_norm):
    """!
    Calculate estimates for beam parameters from an input field component.
    These estimates are used as initial values for Gaussian fitting.

    @param x Grid of x co-ordinates of input field.
    @param y Grid of y co-ordinates of input field.
    @param field_norm Normalised input field component.

    @returns x0 Semi-major axis size of estimate.
    @returns y0 Semi-minor axis size of estimate.
    @returns xm Mean x-center of estimate.
    @returns ym Mean y-center of estimate.
    @returns theta Position angle of estimate.
    """

    M0 = np.nansum(field_norm)
    xm = np.nansum(x * field_norm) / M0
    ym = np.nansum(y * field_norm) / M0

    Mxx = np.nansum(x**2 * field_norm) / M0 - xm**2
    Myy = np.nansum(y**2 * field_norm) / M0 - ym**2
    Mxy = np.nansum(x*y * field_norm) / M0 - xm*ym

    D = 1 / (2 * (Mxx*Myy - Mxy**2))

    a = Myy * D 
    b = Mxx * D
    c = -Mxy * D

    Amp = 1 #M0 * np.sqrt(a * b - c**2) * area[0,0]
    theta = 0.5 * np.arctan(2 * c / (a - b))

    if a-b > 0: # Not HW1 but the largest axis corresponds to theta.
        theta += np.pi/2
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

def fitGaussAbs(field, surfaceObject, thres, scale, ratio=1):
    """!
    Fit a Gaussian to an amplitude pattern of a field component.
    First, the center and position angle of the pattern is calculated using the method of image moments.
    Then, local maxima that might interfere with fitting are removed using the findConnectedSubsets method from MatUtils.py.

    @param field Component of field to fit.
    @param surfaceObject Surface on which the field is defined.
    @param thres Threshold for fitting in decibels.
    @param mode Whether to fit the Gaussian in linear or logarithmic space.
    @param ratio Allowed maximal ratio of fit to actual beam. If "None", will just attempt to fit the Gaussian to supplied pattern. 
            If given, will only accept a fit if the ratio of integrated power in the fitted Gaussian to the supplied beam pattern is less 
            than or equal to the given value. Defaults to 1.

    @returns popt Optimal parameters for Gaussian.
    """

    global thres_g
    thres_g = thres

    grids = BRefl.generateGrid(surfaceObject, transform=False, spheric=False)

    if surfaceObject["gmode"] == 1:
        cp_field = deepcopy(field)
        pr_x = grids.x
        pr_y = grids.y

        x, y = np.mgrid[np.nanmin(pr_x):np.nanmax(pr_x):1j*surfaceObject["gridsize"][0],
                        np.nanmin(pr_y):np.nanmax(pr_y):1j*surfaceObject["gridsize"][1]]

        reg_field = griddata((pr_x.ravel(), pr_y.ravel()), 
                             np.absolute(field).ravel(),
                             (x, y))

        import matplotlib.pyplot as plt
        plt.imshow(reg_field)
        plt.show()

        _field = np.absolute(reg_field) / np.nanmax(np.absolute(reg_field))

    else:
        _field = np.absolute(field) / np.nanmax(np.absolute(field))
        x = grids.x
        y = grids.y

    if scale == Scales.dB:
        fit_field = 20 * np.log10(_field)
        mask_f = fit_field >= thres 

    elif scale == Scales.LIN:
        fit_field = _field
        mask_f = fit_field >= 10**(thres/20)

    idx_max = np.unravel_index(np.nanargmax(fit_field), fit_field.shape)

    x_max = x[idx_max]
    y_max = y[idx_max] 

    idx_rows, idx_cols = MUtils.findConnectedSubsets(mask_f, 1, idx_max)
    _xmin = x[np.nanmin(idx_cols), idx_max[0]]
    _xmax = x[np.nanmax(idx_cols), idx_max[0]]
   
    _ymin = y[idx_max[1], np.nanmin(idx_rows)]
    _ymax = y[idx_max[1], np.nanmax(idx_rows)]
    
    x_cond = (x > _xmin) & (x < _xmax)
    y_cond = (y > _ymin) & (y < _ymax)
    _mask_f = mask_f & x_cond & y_cond
    
    xmask = x[mask_f] - x_max
    ymask = y[mask_f] - y_max

    # Guard: if _mask_f is empty, use no mask
    if not _mask_f.any():
        _mask_f = np.ones(_mask_f.shape, dtype=int)
    
    x0, y0, xs, ys, theta = calcEstimates(x[_mask_f], y[_mask_f], fit_field[_mask_f])

    p0 = [x0, y0, xs, ys, theta, np.max(fit_field)]

    _ratio = ratio
    num = 0

    if ratio is None:
        args = (surfaceObject, fit_field, _mask_f, scale)
        popt = fmin(GaussAbs, p0, args, disp=False)

    else:
        while _ratio >= ratio:
            if scale == Scales.dB:
                fit_field = 20 * np.log10(_field)
                mask_f = fit_field >= thres 

            elif scale == Scales.LIN:
                fit_field = _field
                mask_f = fit_field >= 10**(thres/20)

            if num >= 1:
                _mask_f = mask_f & x_cond & y_cond
            else:
                _mask_f = mask_f

            args = (surfaceObject, fit_field, _mask_f, scale)
            popt = fmin(GaussAbs, p0, args, disp=False)

            _Psi = generateGauss(popt, surfaceObject, scale=Scales.LIN)
            _ratio = np.nansum(_Psi**2) / np.nansum(_field**2)
            
            if num > 1:
                if thres < -3:
                    thres += 0.5
            
            num += 1

    return popt

def GaussAbs(p0, *args):
    """!
    Generate absolute Gaussian from parameters.
    Called in optimalisation. This method returns an overlap parameter of the fitted Gaussian .

    @param p0 Array containing parameters for Gaussian.
    @param args Extra arguments for defining Gaussian and fit.

    @returns epsilon Coupling of field with Gaussian.
    """

    surfaceObject, field_est, mask, scale = args
    Psi = generateGauss(p0, surfaceObject, scale)
    coup = np.sum(np.absolute(Psi[mask])**2) / np.sum(np.absolute(field_est[mask])**2)
    epsilon = np.absolute(1 - coup)
    
    num = np.sum(field_est[mask] * Psi[mask])**2
    normE = np.sum(np.absolute(field_est[mask])**2)
    normP = np.sum(np.absolute(Psi[mask])**2)

    c00 = num / normE / normP
    
    eta = np.absolute(c00)
    epsilon = np.absolute(1 - eta)
    
    return epsilon

def generateGauss(p0, surfaceObject, scale):
    """!
    Generate a Gaussian from Gaussian and surface parameters.

    @param p0 Gaussian parameters.
    @param surfaceObject Surface on which Gaussian is defined.
    @param scale Whether to generate Gaussian in linear or decibel space.

    @returns Psi Gaussian distribution.
    """

    x0, y0, xs, ys, theta, amp = p0
    
    grids = BRefl.generateGrid(surfaceObject, transform=False, spheric=False)
    x = grids.x
    y = grids.y
    
    a = np.cos(theta)**2 / (2 * x0**2) + np.sin(theta)**2 / (2 * y0**2)
    c = np.sin(2 * theta) / (4 * x0**2) - np.sin(2 * theta) / (4 * y0**2)
    b = np.sin(theta)**2 / (2 * x0**2) + np.cos(theta)**2 / (2 * y0**2)

    if scale == Scales.dB:
        Psi = 20*np.log10(np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2)))

    elif scale == Scales.LIN:
        Psi = np.exp(-(a*(x - xs)**2 + 2*c*(x - xs)*(y - ys) + b*(y - ys)**2))
    
    return Psi.reshape(x.shape)
