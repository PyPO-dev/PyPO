import numpy as np
from PyPO.PyPOTypes import *
from PyPO.BindRefl import *
from PyPO.FitGauss import fitGaussAbs

##
# @file
# File containing functions for calculating efficiencies and other metrics.

##
# Generate an elliptical mask for spillover and taper efficiency calculations.
# The ellipse has outer and inner axes and can represent a mask that is open in the center.
# The mask is generated in the xy plane.
#
# @param x Grid of x co-ordinates of surface for mask.
# @param y Grid of y co-ordinates of surface for mask.
# @param aperDict Dictionary containing ellipse parameters of mask.
#
# @returns mask The mask.
def _generateMask(x, y, aperDict):
    
    t = np.arctan2(y, x) + np.pi

    outer = (aperDict["outer"][0] * np.cos(t))**2 + (aperDict["outer"][1] * np.sin(t))**2
    inner = (aperDict["inner"][0] * np.cos(t))**2 + (aperDict["inner"][1] * np.sin(t))**2

    cond1 = (x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2 < outer
    cond2 = (x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2 > inner
 
    return cond1 & cond2

##
# Calculate the mean geometric center of a ray-trace frame.
#
# @param frame Frame to calculate center of.
#
# @returns center Array containing center xyz co-ordinates.
def calcRTcenter(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    c_x = np.sum(frame.x[idx_good]) / len(frame.x[idx_good])
    c_y = np.sum(frame.y[idx_good]) / len(frame.y[idx_good])
    c_z = np.sum(frame.z[idx_good]) / len(frame.z[idx_good])
    
    return np.array([c_x, c_y, c_z])

##
# Calculate mean tilt of a ray-trace frame.
#
# @param frame Frame to calculate tilt of.
#
# @returns tilt Array containing xyz tilt components.
def calcRTtilt(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    t_x = np.sum(frame.dx[idx_good]) / len(frame.dx[idx_good])
    t_y = np.sum(frame.dy[idx_good]) / len(frame.dy[idx_good])
    t_z = np.sum(frame.dz[idx_good]) / len(frame.dz[idx_good])
    
    return np.array([t_x, t_y, t_z]) / np.linalg.norm(np.array([t_x, t_y, t_z]))

##
# Calculate standard deviation tilt of a ray-trace frame.
#
# @param frame Frame to calculate standard deviation tilt of.
#
# @returns tilt Array containing xyz tilt standard deviations components.
def calcRTtiltSTD(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    stdt_x = np.std(np.absolute(frame.dx[idx_good]))
    stdt_y = np.std(np.absolute(frame.dy[idx_good]))
    stdt_z = np.std(np.absolute(frame.dz[idx_good]))

    return np.array([stdt_x, stdt_y, stdt_z])# / np.linalg.norm(np.array([stdt_x, stdt_y, stdt_z]))

##
# Calculate root-mean-square (RMS) of a ray-trace frame.
#
# @param frame Frame to calculate RMS of.
#
# @returns rms The RMS value of the frame.
def calcRMS(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    c_f = calcRTcenter(frame) 
    rms = np.sqrt(np.sum((frame.x[idx_good] - c_f[0])**2 + (frame.y[idx_good] - c_f[1])**2 + (frame.z[idx_good] - c_f[2])**2) / len(frame.x[idx_good]))

    return rms

##
# Calculate spillover efficiency of a field on an aperture.
#
# @param field Component of a fields object.
# @param surfaceObject Name of surface on which field is defined.
# @param aperDict Dictionary containing parameters for aperture for calculation.
#
# @returns eff_s Spillover efficiency
def calcSpillover(field, surfaceObject, aperDict):
    # Generate the grid in restframe
    grids = generateGrid(surfaceObject, transform=False, spheric=True)

    x = grids.x
    y = grids.y
    area = grids.area
    mask = _generateMask(x, y, aperDict) 
    field_ap = field * mask.astype(complex)
    area_m = area * mask.astype(int)
    eff_s = np.absolute(np.sum(np.conj(field_ap) * field * area))**2 / (np.sum(np.absolute(field)**2 * area) * np.sum(np.absolute(field_ap)**2 *area_m))

    if np.isnan(eff_s):
        eff_s = 0

    return eff_s

##
# Calculate taper efficiency of a field on an aperture.
#
# @param field Component of a fields object.
# @param surfaceObject Name of surface on which field is defined.
# @param aperDict Dictionary containing parameters for aperture for calculation.
#
# @returns eff_t Taper efficiency
def calcTaper(field, surfaceObject, aperDict):
    grids = generateGrid(surfaceObject, transform=False, spheric=True)
    area = grids.area
    
    if aperDict:
        x = grids.x
        y = grids.y
        mask = _generateMask(x, y, aperDict) 


        field = field[mask]
        area = area[mask]

    eff_t = np.absolute(np.sum(field * area))**2 / np.sum(np.absolute(field)**2 * area) / np.sum(area)
    
    if np.isnan(eff_t):
        eff_t = 0

    return eff_t

##
# Calculate cross-polar efficiency.
#
# @param Cofield Co-polarised field component.
# @param Xfield Cross-polarised field component.
#
# @returns eff_Xpol Cross-polar efficiency.
def calcXpol(Cofield, Xfield):
    eff_Xpol = 1 - np.sum(np.absolute(Xfield)**2) / (np.sum(np.absolute(Cofield)**2)+np.sum(np.absolute(Xfield)**2))

    return eff_Xpol

##
# Calculate main beam efficiency.
#
# @param field Component of a fields object.
# @param surfaceObject Name of surface on which field is defined.
# @param fitGauss Gaussian fit to field.
#
# @returns eff_mb Main beam efficiency.
def calcMainBeam(field, surfaceObject, fitGauss):
    field_norm = field / np.max(np.absolute(field))
    fitGauss_norm = fitGauss / np.max(np.absolute(fitGauss))
    
    eff_mb = np.sum(np.absolute(fitGauss_norm)**2) / np.sum(np.absolute(field_norm)**2)
    
    return eff_mb
