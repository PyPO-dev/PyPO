import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


import PyPO.Checks as PChecks
import PyPO.Structs as PStructs
import PyPO.BindUtils as BUtils
import PyPO.BindBeam as BBeam
import PyPO.BindRefl as BRefl
from PyPO.CustomLogger import CustomLogger

import logging

import ctypes as ct

from PyPO.System import System
from PyPO.Enums import FieldComponents, CurrentComponents, Units, Scales

s = System()

# Setting up parameters for defining simulation.
# Source parameters and distances
lam = 1*Units.MM                # Wavelength of light in mm

w0 = 5*lam  # Gaussian beamwaist
z = lam

r_source = 2.5*w0               # Radius of disk in mm
print(f"Radius of source grid     : {r_source:7.3f} mm")

theta_0 = np.rad2deg(np.atan(lam/(np.pi*w0))) # Calculate far-field divergence angle
print(f"Farfield divergence angle : {theta_0:7.3f}°")

fwhm = 1.18*theta_0
print(f"Farfield FWHM angle       : {fwhm:7.3f}°")

z_grid = 10.0
print(f"Near-field grid distance     : {z_grid:7.3f} mm")

w_grid = w0*np.sqrt(1 + ((lam*z_grid)/(np.pi*w0**2))**2)
print(f"Beamwidth at near-field grid : {w_grid:7.3f} mm")

## Calculate the required source grid size for the far-field case

# This estimate is good for about -30-40 dB field accuracy
po_z = 1.09*np.pi*2*r_source/lam*np.sin(np.deg2rad(theta_0*5)) + 10

po1 = int(po_z/2.4)
po2 = int(po_z)

max_po1 = int(2*r_source/lam)
max_po2 = int(4*np.pi*r_source/lam)

print(f"Far-Field Case:")
print(f"Estimated Grid size : ({po1:d}, {po2:d})")
print(f"Maximum Grid size   : ({max_po1:d}, {max_po2:d})")

# Setting up surface dictionaries and source current distributions.
source_plane = {
        "name"      : "source_plane",
        "gmode"     : "uv",
        # "lims_x"    : np.array([-r_disk, r_disk]),
        # "lims_y"    : np.array([-r_disk, r_disk]),
        "lims_u"    : np.array([0, r_source]),
        "lims_v"    : np.array([0, 360])*Units.DEG,
        "gridsize"  : np.array([max_po1, max_po2])
        }

s.addPlane(source_plane)

source_grid = s.generateGrids("source_plane")

vecGPODict = {
        "name"  : "source_gb",
        "lam"   : lam,
        "w0"    : w0,
        "z"     : z,
        "n"     : 1.0,
        "power" : np.pi*4
}

GPODict = {
        "name"  : "source_g",
        "lam"   : lam,
        "w0x"   : w0,
        "w0y"   : w0,
        "n"     : 1.0,
        "dxyx"  : 0.0,
        "E0"    : np.sqrt(np.pi*4),
        "pol"   : np.array([1, 0, 0])
}

# s.createGaussian(GPODict, 'source_plane')

s.createGaussianBeam(vecGPODict, 'source_plane')