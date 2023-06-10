import numpy as np
import PyPO.System as System

##
# @file
# This file contains commonly used input templates so that we do not have to type them over and over again.

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
        "lam"       : 1,
        "n"         : 1,
        "seed"      : 1,
        "x0"        : 1,
        "y0"        : 1
        }

GPOfield =  {
        "name"      : "testGaussField",
        "lam"       : 1,
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

#### PARABOLOIDS ####
paraboloid_man_xy = {
        "name"      : "testParaboloid_man_xy",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1]),
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

paraboloid_man_uv = {
        "name"      : "testParaboloid_man_uv",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1]),
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
        }

paraboloid_foc_xy = {
        "name"      : "testParaboloid_foc_xy",
        "pmode"     : "focus",
        "vertex"    : np.array([0, 0, 0]),
        "focus_1"   : np.array([0, 0, 1]),
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

paraboloid_foc_uv = {
        "name"      : "testParaboloid_foc_uv",
        "pmode"     : "focus",
        "vertex"    : np.array([0, 0, 0]),
        "focus_1"   : np.array([0, 0, 1]),
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
        }

hyperboloid_man_uv = {
        "name"      : "testHyperboloid_man_uv",
        "pmode"     : "manual",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 1.1,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
        }

#### ELLIPSOIDS ####
ellipsoid_z_man_xy = {
        "name"      : "testEllipsoid_z_man_xy",
        "pmode"     : "manual",
        "orient"    : "z",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

ellipsoid_z_man_uv = {
        "name"      : "testEllipsoid_z_man_uv",
        "pmode"     : "manual",
        "orient"    : "z",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
        }

ellipsoid_x_man_xy = {
        "name"      : "testEllipsoid_x_man_xy",
        "pmode"     : "manual",
        "orient"    : "x",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 0.5,
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

ellipsoid_x_man_uv = {
        "name"      : "testEllipsoid_x_man_uv",
        "pmode"     : "manual",
        "orient"    : "x",
        "coeffs"    : np.array([1, 1, 1]),
        "ecc"       : 0.5,
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
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
        "gridsize"  : np.array([31, 31])
        }

#### PLANES ####
plane_xy = {
        "name"      : "testPlane_xy",
        "gmode"     : "xy",
        "lims_x"    : np.array([-1, 1]),
        "lims_y"    : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

plane_uv = {
        "name"      : "testPlane_uv",
        "gmode"     : "uv",
        "lims_u"    : np.array([0, 1]),
        "lims_v"    : np.array([0, 360]),
        "gridsize"  : np.array([31, 31])
        }

plane_AoE = {
        "name"      : "testPlane_AoE",
        "gmode"     : "AoE",
        "lims_Az"   : np.array([-1, 1]),
        "lims_El"   : np.array([-1, 1]),
        "gridsize"  : np.array([31, 31])
        }

aperDict = {
        "plot"      : False,
        "center"    : np.array([0, 0]),
        "outer"     : np.array([0.5, 0.5]),
        "inner"     : np.array([0, 0])
        }

##
# Get a system with all possible reflectortypes.
def getSystemWithReflectors():
    s = System(verbose=False)

    s.addPlane(TestTemplates.plane_xy)
    s.addPlane(TestTemplates.plane_uv)
    s.addPlane(TestTemplates.plane_AoE)

    s.addParabola(TestTemplates.paraboloid_man_xy)
    s.addParabola(TestTemplates.paraboloid_man_uv)
    s.addParabola(TestTemplates.paraboloid_foc_xy)
    s.addParabola(TestTemplates.paraboloid_foc_uv)

    s.addHyperbola(TestTemplates.hyperboloid_man_xy)
    s.addHyperbola(TestTemplates.hyperboloid_man_uv)
    s.addHyperbola(TestTemplates.hyperboloid_foc_xy)
    s.addHyperbola(TestTemplates.hyperboloid_foc_uv)

    s.addEllipse(TestTemplates.ellipsoid_x_man_xy)
    s.addEllipse(TestTemplates.ellipsoid_x_man_uv)
    s.addEllipse(TestTemplates.ellipsoid_x_foc_xy)
    s.addEllipse(TestTemplates.ellipsoid_x_foc_uv)
    s.addEllipse(TestTemplates.ellipsoid_z_man_xy)
    s.addEllipse(TestTemplates.ellipsoid_z_man_uv)
    s.addEllipse(TestTemplates.ellipsoid_z_foc_xy)
    s.addEllipse(TestTemplates.ellipsoid_z_foc_uv)

    return s
