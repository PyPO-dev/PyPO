import unittest
import numpy as np
import ctypes

import TestUtils.TestTemplates as TestTemplates

import PyPO.BindRefl as refllibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if grids in PyPO are correct

class Test_SystemGrids(unittest.TestCase):
    def setUp(self):
        self.s = System(verbose=False)

    def test_loadRefllib(self):
        lib = refllibs.loadRefllib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_generateGridsPlane(self):
        self.s.addPlane(TestTemplates.plane_xy)
        grids = self.s.generateGrids("testPlane_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addPlane(TestTemplates.plane_uv)
        grids = self.s.generateGrids("testPlane_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addPlane(TestTemplates.plane_AoE)
        grids = self.s.generateGrids("testPlane_AoE")
        self.assertEqual(type(grids), pypotypes.reflGrids)
        
    def test_generateGridsParaboloid(self):
        self.s.addParabola(TestTemplates.paraboloid_man_xy)
        grids = self.s.generateGrids("testParaboloid_man_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addParabola(TestTemplates.paraboloid_man_uv)
        grids = self.s.generateGrids("testParaboloid_man_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addParabola(TestTemplates.paraboloid_foc_xy)
        grids = self.s.generateGrids("testParaboloid_foc_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addParabola(TestTemplates.paraboloid_foc_uv)
        grids = self.s.generateGrids("testParaboloid_foc_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)
    
    def test_generateGridsHyperboloid(self):
        self.s.addHyperbola(TestTemplates.hyperboloid_man_xy)
        grids = self.s.generateGrids("testHyperboloid_man_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addHyperbola(TestTemplates.hyperboloid_man_uv)
        grids = self.s.generateGrids("testHyperboloid_man_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addHyperbola(TestTemplates.hyperboloid_foc_xy)
        grids = self.s.generateGrids("testHyperboloid_foc_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addHyperbola(TestTemplates.hyperboloid_foc_uv)
        grids = self.s.generateGrids("testHyperboloid_foc_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)
    
    def test_generateGridsEllipsoid(self):
        self.s.addEllipse(TestTemplates.ellipsoid_x_man_xy)
        grids = self.s.generateGrids("testEllipsoid_x_man_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_x_man_uv)
        grids = self.s.generateGrids("testEllipsoid_x_man_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_x_foc_xy)
        grids = self.s.generateGrids("testEllipsoid_x_foc_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_x_foc_uv)
        grids = self.s.generateGrids("testEllipsoid_x_foc_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)
        
        self.s.addEllipse(TestTemplates.ellipsoid_z_man_xy)
        grids = self.s.generateGrids("testEllipsoid_z_man_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_z_man_uv)
        grids = self.s.generateGrids("testEllipsoid_z_man_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_z_foc_xy)
        grids = self.s.generateGrids("testEllipsoid_z_foc_xy")
        self.assertEqual(type(grids), pypotypes.reflGrids)

        self.s.addEllipse(TestTemplates.ellipsoid_z_foc_uv)
        grids = self.s.generateGrids("testEllipsoid_z_foc_uv")
        self.assertEqual(type(grids), pypotypes.reflGrids)
if __name__ == "__main__":
    unittest.main()
