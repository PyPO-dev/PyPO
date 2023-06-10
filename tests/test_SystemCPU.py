import unittest
import numpy as np
import ctypes

import TestUtils.TestTemplates as TestTemplates

import PyPO.BindCPU as cpulibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if beams in PyPO are correct

class Test_SystemCPU(unittest.TestCase):
    def setUp(self):
        self.s = System(verbose=False)
        
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.addPlane(TestTemplates.plane_uv)
        self.s.addPlane(TestTemplates.plane_AoE)
        
        self.s.addParabola(TestTemplates.paraboloid_man_xy)
        self.s.addParabola(TestTemplates.paraboloid_man_uv)
        self.s.addParabola(TestTemplates.paraboloid_foc_xy)
        self.s.addParabola(TestTemplates.paraboloid_foc_uv)

        self.s.addHyperbola(TestTemplates.hyperboloid_man_xy)
        self.s.addHyperbola(TestTemplates.hyperboloid_man_uv)
        self.s.addHyperbola(TestTemplates.hyperboloid_foc_xy)
        self.s.addHyperbola(TestTemplates.hyperboloid_foc_uv)
        
        self.s.addEllipse(TestTemplates.ellipsoid_x_man_xy)
        self.s.addEllipse(TestTemplates.ellipsoid_x_man_uv)
        self.s.addEllipse(TestTemplates.ellipsoid_x_foc_xy)
        self.s.addEllipse(TestTemplates.ellipsoid_x_foc_uv)
        self.s.addEllipse(TestTemplates.ellipsoid_z_man_xy)
        self.s.addEllipse(TestTemplates.ellipsoid_z_man_uv)
        self.s.addEllipse(TestTemplates.ellipsoid_z_foc_xy)
        self.s.addEllipse(TestTemplates.ellipsoid_z_foc_uv)

    def test_loadCPUlib(self):
        lib = cpulibs.loadCPUlib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_runPO(self):
        s.addPlane()


if __name__ == "__main__":
    unittest.main()

