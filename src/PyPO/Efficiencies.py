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

def calcRMS(frame):
    c_x = np.sum(frame.x) / len(frame.x)
    c_y = np.sum(frame.y) / len(frame.y)
    c_z = np.sum(frame.z) / len(frame.z)
    rms = np.sqrt(np.sum((frame.x - c_x)**2 + (frame.y - c_y)**2 + (frame.z - c_z)**2) / len(frame.x))

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
        cond1 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) < aperDict["r_out"]
        cond2 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) > aperDict["r_in"]
        mask = cond1 & cond2


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
