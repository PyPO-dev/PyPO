import unittest
import numpy as np
import ctypes

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import PyPO.BindRefl as refllibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if grids in PyPO are correct

class Test_SystemGrids(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

    def test_loadRefllib(self):
        lib = refllibs.loadRefllib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_generateGridsPlane(self):
        for plane in TestTemplates.getPlaneList():
            grids = self.s.generateGrids(plane["name"])
            self.assertEqual(type(grids), pypotypes.reflGrids)
        
    def test_generateGridsParaboloid(self):
        for parabola in TestTemplates.getPlaneList():
            grids = self.s.generateGrids(parabola["name"])
            self.assertEqual(type(grids), pypotypes.reflGrids)
    
    def test_generateGridsHyperboloid(self):
        for hyperbola in TestTemplates.getPlaneList():
            grids = self.s.generateGrids(hyperbola["name"])
            self.assertEqual(type(grids), pypotypes.reflGrids)
    
    def test_generateGridsEllipsoid(self):
        for ellipse in TestTemplates.getPlaneList():
            grids = self.s.generateGrids(ellipse["name"])
            self.assertEqual(type(grids), pypotypes.reflGrids)

if __name__ == "__main__":
    unittest.main()
