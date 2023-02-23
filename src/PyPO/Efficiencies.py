import numpy as np
import matplotlib.pyplot as pt
from src.PyPO.PyPOTypes import *
from src.PyPO.BindRefl import *

def _generateMask(x, y, aperDict):
    
    t = np.arctan2(y, x) + np.pi

    outer = (aperDict["outer"][0] * np.cos(t))**2 + (aperDict["outer"][1] * np.sin(t))**2
    inner = (aperDict["inner"][0] * np.cos(t))**2 + (aperDict["inner"][1] * np.sin(t))**2

    cond1 = (x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2 < outer
    cond2 = (x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2 > inner
 
    return cond1 & cond2

def calcRTcenter(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    c_x = np.sum(frame.x[idx_good]) / len(frame.x[idx_good])
    c_y = np.sum(frame.y[idx_good]) / len(frame.y[idx_good])
    c_z = np.sum(frame.z[idx_good]) / len(frame.z[idx_good])
    
    return np.array([c_x, c_y, c_z])

def calcRTtilt(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    t_x = np.sum(frame.dx[idx_good]) / len(frame.dx[idx_good])
    t_y = np.sum(frame.dy[idx_good]) / len(frame.dy[idx_good])
    t_z = np.sum(frame.dz[idx_good]) / len(frame.dz[idx_good])
    
    return np.array([t_x, t_y, t_z])

def calcRMS(frame):
    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)
    c_f = calcRTcenter(frame) 
    rms = np.sqrt(np.sum((frame.x[idx_good] - c_f[0])**2 + (frame.y[idx_good] - c_f[1])**2 + (frame.z[idx_good] - c_f[2])**2) / len(frame.x[idx_good]))

    return rms

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

    return eff_s

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

    return eff_t

def calcXpol(Cofield, Xfield):
    eff_Xpol = 1 - np.sum(np.absolute(Xfield)**2) / (np.sum(np.absolute(Cofield)**2)+np.sum(np.absolute(Xfield)**2))

    return eff_Xpol

def calcDirectivity(eta_t, surfaceObject, k):
    grids = generateGrid(surfaceObject, transform=False, spheric=True)

    D = 10*np.log10(k**2 / np.pi * eta_t * np.sum(grids.area))

    return D


if __name__ == "__main__":
    print("Functions to calculate system efficiencies.")
