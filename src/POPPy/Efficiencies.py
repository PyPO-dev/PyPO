import numpy as np

from src.POPPy.POPPyTypes import *
from src.POPPy.BindRefl import *

def calcSpillover(field, surfaceObject, aperDict):
    # Generate the grid in restframe
    grids = generateGrid(surfaceObject, transform=False, spheric=True)

    x = grids.x
    y = grids.y
    area = grids.area

    cond1 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) < aperDict["r_out"]
    cond2 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) > aperDict["r_in"]
    mask = cond1 & cond2

    field_ap = field * mask.astype(complex)
    eff_s = np.absolute(np.sum(np.conj(field_ap) * field)**2) / (np.sum(np.absolute(field)**2) * np.sum(np.absolute(field_ap)**2))

    return eff_s

def calcTaper(field, surfaceObject, aperDict):
    grids = generateGrid(surfaceObject, transform=False, spheric=True)

    x = grids.x
    y = grids.y
    area = grids.area

    cond1 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) < aperDict["r_out"]
    cond2 = np.sqrt((x - aperDict["center"][0])**2 + (y - aperDict["center"][1])**2) > aperDict["r_in"]
    mask = cond1 & cond2


    field_ap = field[mask]
    area_m = area[mask]

    eff_t = np.absolute(np.sum(field_ap * area_m))**2 / np.sum(np.absolute(field_ap)**2 * area_m) / np.sum(area_m)

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
