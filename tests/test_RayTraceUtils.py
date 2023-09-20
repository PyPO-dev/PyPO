"""!
@file
Script for testing the ray-trace functionalities of PyPO.
"""

from scipy.stats import special_ortho_group

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import unittest
import numpy as np

from PyPO.System import System
import PyPO.MatTransform as mt

class Test_RayTraceUtils(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

    def test_focusFind(self):
        focus = self.s.findRTfocus(TestTemplates.TubeRTframe["name"])
        self.assertTrue(focus.shape == (3,))
    
    def test_findRotation(self):
        R = special_ortho_group.rvs(dim=3)

        v = np.array([1, 1, 1])

        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 0, 1])

        u = R @ v

        R_find = self.s.findRotation(v, u)
        _u = R_find[:-1, :-1] @ v
        
        for ri, ro in zip(u, _u):
            self.assertAlmostEqual(ri, ro)
    
    def test_getAnglesFromMatrix(self):
        R = special_ortho_group.rvs(dim=3)

        angles = self.s.getAnglesFromMatrix(R)
        
        R_find = mt.MatRotate(angles)
        
        v = np.array([1, 1, 1])

        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 0, 1])

        u = R @ v
        _u = R_find[:-1, :-1] @ v

        for ri, ro in zip(u, _u):
            self.assertAlmostEqual(ri, ro)
            
if __name__ == "__main__":
    import nose2
    nose2.main()
