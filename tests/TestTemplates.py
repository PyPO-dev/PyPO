"""!
@file
This file contains commonly used input templates so that we do not have to type them over and over again.
"""

import numpy as np
from PyPO.System import System
from PyPO.Enums import AperShapes

TubeRTframe =  {
        "name"      : "testTubeRTframe",
        "nRays"     : 7,
        "nRing"     : 7,
        "angx0"     : 1,
        "angy0"     : 1,
        "x0"        : 1,
        "y0"        : 1
        }

GaussRTframe =  {
        "name"      : "testGaussRTframe",
        "nRays"     : 7,
        "lam"       : 0.01,
        "n"         : 1,
        "seed"      : 1,
        "x0"        : 1,
        "y0"        : 1
        }

GPOfield =  {
        "name"      : "testGaussField",
        "lam"       : 0.01,
        "w0x"       : 1,
        "w0y"       : 1,
        "n"         : 1,
        "E0"        : 1,
        "dxyz"      : 1,
        "pol"       : np.array([1,0,0])
        }

PS_Ufield =  {
        "name"      : "testPS_UField",
        "lam"       : 1,
        "E0"        : 1,
        "pol"       : np.array([1,0,0])
        }

PS_Ufield_FF =  {
        "name"      : "testPS_UField_FF",
        "lam"       : 1,
        "E0"        : 1,
        "pol"       : np.array([1,0,0])
        }

#### PARABOLOIDS ####
paraboloid_man_xy = {
        "name"      : "testParaboloid_man_xy",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1]),
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

paraboloid_man_uv = {
        "name"      : "testParaboloid_man_uv",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1]),
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

paraboloid_foc_xy = {
        "name"      : "testParaboloid_foc_xy",
        "pmode"     : "focus",
        "vertex"    : np.array([0, 0, 0]),
        "focus_1"   : np.array([0, 0, 1]),
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

paraboloid_foc_uv = {
        "name"      : "testParaboloid_foc_uv",
        "pmode"     : "focus",
        "vertex"    : np.array([0, 0, 0]),
        "focus_1"   : np.array([0, 0, 1]),
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

#### HYPERBOLOIDS ####
hyperboloid_man_xy = {
        "name"      : "testHyperboloid_man_xy",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 1.1,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

hyperboloid_man_uv = {
        "name"      : "testHyperboloid_man_uv",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 1.1,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

hyperboloid_foc_xy = {
        "name"      : "testHyperboloid_foc_xy",
        "pmode"     : "focus",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 1.1,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

hyperboloid_foc_uv = {
        "name"      : "testHyperboloid_foc_uv",
        "pmode"     : "focus",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 1.1,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

#### ELLIPSOIDS ####
ellipsoid_z_man_xy = {
        "name"      : "testEllipsoid_z_man_xy",
        "pmode"     : "manual",
        "orient"    : "z",
        "coeffs"    : np.array([10, 10, 1]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

ellipsoid_z_man_uv = {
        "name"      : "testEllipsoid_z_man_uv",
        "pmode"     : "manual",
        "orient"    : "z",
        "coeffs"    : np.array([10, 10, 1]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

ellipsoid_z_foc_xy = {
        "name"      : "testEllipsoid_z_foc_xy",
        "pmode"     : "focus",
        "orient"    : "z",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

ellipsoid_z_foc_uv = {
        "name"      : "testEllipsoid_z_foc_uv",
        "pmode"     : "focus",
        "orient"    : "z",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

ellipsoid_x_man_xy = {
        "name"      : "testEllipsoid_x_man_xy",
        "pmode"     : "manual",
        "orient"    : "x",
        "coeffs"    : np.array([1, 10, 10]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-0.1, 0.1]),
        "lims_y"    : np.array([-0.1, 0.1]),
        "gridsize"  : np.array([13, 13])
        }

ellipsoid_x_man_uv = {
        "name"      : "testEllipsoid_x_man_uv",
        "pmode"     : "manual",
        "orient"    : "x",
        "coeffs"    : np.array([1, 10, 10]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 0.1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

ellipsoid_x_foc_xy = {
        "name"      : "testEllipsoid_x_foc_xy",
        "pmode"     : "focus",
        "orient"    : "x",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

ellipsoid_x_foc_uv = {
        "name"      : "testEllipsoid_x_foc_uv",
        "pmode"     : "focus",
        "orient"    : "x",
        "focus_1"   : np.array([0, 0, 1]),
        "focus_2"   : np.array([0, 0, -1]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([13, 12])
        }

#### PLANES ####
plane_xy = {
        "name"      : "testPlane_xy",
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

plane_uv = {
        "name"      : "testPlane_uv",
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([103, 102])
        }

plane_AoE = {
        "name"      : "testPlane_AoE",
        "gmode"     : "AoE",
        "lims_Az"   : np.array([-1, 1]),
        "lims_El"   : np.array([-1, 1]),
        "gridsize"  : np.array([13, 13])
        }

aperDictEll = {
        "shape"     : AperShapes.ELL,
        "plot"      : False,
        "center"    : np.array([0, 0]),
        "outer"     : np.array([0.5, 0.5]),
        "inner"     : np.array([0, 0])
        }

aperDictRect = {
        "shape"     : AperShapes.RECT,
        "plot"      : False,
        "center"    : np.array([0, 0]),
        "outer_x"   : np.array([-0.5, 0.5]),
        "inner_x"   : np.array([-0.2, 0.2]),
        "outer_y"   : np.array([-0.5, 0.5]),
        "inner_y"   : np.array([-0.2, 0.2]),
        }

##
# Get a list of plane dictionaries.
#
# @returns out List of all plane dictionaries
def getPlaneList():
    out = [plane_uv, plane_xy, plane_AoE]
    return out

##
# Get a list of reflector plane dictionaries. No far-field plane.
#
# @returns out List of all plane dictionaries
def getReflectorPlaneList():
    out = [plane_uv, plane_xy]
    return out

##
# Get a list of paraboloid dictionaries.
#
# @returns out List of all paraboloid dictionaries
def getParaboloidList():
    out = [paraboloid_man_xy, paraboloid_man_uv, 
            paraboloid_foc_xy, paraboloid_foc_uv]
    return out

##
# Get a list of hyperboloid dictionaries.
#
# @returns out List of all hyperboloid dictionaries
def getHyperboloidList():
    out = [hyperboloid_man_xy, hyperboloid_man_uv, 
            hyperboloid_foc_xy, hyperboloid_foc_uv]
    return out

##
# Get a list of ellipsoid dictionaries.
#
# @returns out List of all ellipsoid dictionaries
def getEllipsoidList():
    out = [ellipsoid_x_man_xy, ellipsoid_x_man_uv, 
            ellipsoid_x_foc_xy, ellipsoid_x_foc_uv,
            ellipsoid_z_man_xy, ellipsoid_z_man_uv,
            ellipsoid_z_foc_xy, ellipsoid_z_foc_uv
            ]
    return out

##
# Get all surfaces
def getAllSurfList():
    out = []
    out.extend(getPlaneList())
    out.extend(getParaboloidList())
    out.extend(getHyperboloidList())
    out.extend(getEllipsoidList())
    
    return out

##
# Get all reflectors
def getAllReflectorList():
    out = [plane_uv, plane_xy,
            paraboloid_man_xy, paraboloid_man_uv, 
            paraboloid_foc_xy, paraboloid_foc_uv,
            hyperboloid_man_xy, hyperboloid_man_uv, 
            hyperboloid_foc_xy, hyperboloid_foc_uv,
            ellipsoid_x_man_xy, ellipsoid_x_man_uv, 
            ellipsoid_x_foc_xy, ellipsoid_x_foc_uv,
            ellipsoid_z_man_xy, ellipsoid_z_man_uv,
            ellipsoid_z_foc_xy, ellipsoid_z_foc_uv
            ]
    return out

##
# Get a list containing all frame dictionaries
#
# @returns out List of all frame dictionaries. 
def getFrameList():
    out = [TubeRTframe, GaussRTframe]
    return out

##
# Get a list containing all PO source dictionaries. Can also be used for scalar sources.
#
# @returns out List of all PO source dictionaries. 
def getPOSourceList():
    out = [GPOfield, PS_Ufield]
    return out

##
# Get a system with all possible reflectortypes.
#
# @returns s System object with all types of reflectors added
def getSystemWithReflectors():
    s = System(verbose=False)

    for plane in getPlaneList():
        s.addPlane(plane)
        if plane["gmode"] != "AoE":
            s.createPointSource(PS_Ufield, plane["name"])
            s.createUniformSource(PS_Ufield, plane["name"])
            s.createGaussian(GPOfield, plane["name"])
            
            s.createPointSourceScalar(PS_Ufield, plane["name"])
            s.createUniformSourceScalar(PS_Ufield, plane["name"])
            s.createScalarGaussian(GPOfield, plane["name"])

        else:
            s.createPointSource(PS_Ufield_FF, plane["name"])

    for parabola in getParaboloidList():
        s.addParabola(parabola)

    for hyperbola in getHyperboloidList():
        s.addHyperbola(hyperbola)

    for ellipse in getEllipsoidList():
        s.addEllipse(ellipse)

    s.createTubeFrame(TubeRTframe)
    s.createGRTFrame(GaussRTframe)
    
    return s
