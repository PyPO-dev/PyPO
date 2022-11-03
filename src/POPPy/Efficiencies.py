import numpy as np

from src.POPPy.POPPyTypes import *
from src.POPPy.BindRefl import *

def calcSpillover(field, surfaceObject, aperDict):
    # Generate the grid in restframe
    grids = generateGrid(surfaceObject, transform=False)

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
    grids = generateGrid(surfaceObject, transform=False)

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

if __name__ == "__main__":
    print("Functions to calculate system efficiencies.")
