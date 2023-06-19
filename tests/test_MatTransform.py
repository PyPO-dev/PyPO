import unittest
import numpy as np

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

from nose2.tools import params

from PyPO.System import System
##
# @file
# 
# Script for testing the transformational formalism of PyPO.

class Test_MatTransform(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False)

        self.funcSelect = {
                0 : self.s.addParabola,
                1 : self.s.addHyperbola,
                2 : self.s.addEllipse,
                3 : self.s.addPlane
                } 

    @params(*TestTemplates.getAllReflectorList())
    def test_translations(self, element):
        par_ref = self.s.copyObj(element)
        par_ref["name"] = par_ref["name"] + "_ref"
        self.funcSelect[self.s.system[element["name"]]["type"]](par_ref)

        name_test = element["name"]
        name_ref = par_ref["name"]
        translation0 = np.array([12, -42, 187])
        translation1 = np.array([-97.3, 237, 66])

        pos_add0 = self.s.copyObj(self.s.system[name_test]["pos"])
        self.s.translateGrids(name_test, translation0)
    
        for x0, x0test in zip(translation0 + pos_add0, self.s.system[name_test]["pos"]):
            self.assertAlmostEqual(x0, x0test, delta=1e-6)
        
        self.s.translateGrids(name_test, translation1)
    
        for x1, x1test in zip(translation1 + translation0 + pos_add0, self.s.system[name_test]["pos"]):
            self.assertAlmostEqual(x1, x1test, delta=1e-6)

        translation0abs = np.random.rand(3)

        self.s.translateGrids(name_test, translation0abs, mode="absolute")
        self.s.translateGrids(name_ref, translation0abs, mode="absolute")

        for x0a, x0atest in zip(translation0abs, self.s.system[name_test]["pos"]):
            self.assertAlmostEqual(x0a, x0atest, delta=1e-6)

        g0atest = self.s.generateGrids(name_test)
        g0aref = self.s.generateGrids(name_ref)

        for x0atest, x0aref in zip(g0atest.x.ravel(), g0aref.x.ravel()):
            self.assertAlmostEqual(x0atest, x0aref, delta=1e-6)
        
        for y0atest, y0aref in zip(g0atest.y.ravel(), g0aref.y.ravel()):
            self.assertAlmostEqual(y0atest, y0aref, delta=1e-6)
        
        for z0atest, z0aref in zip(g0atest.z.ravel(), g0aref.z.ravel()):
            self.assertAlmostEqual(z0atest, z0aref, delta=1e-6)

    @params(*TestTemplates.getAllReflectorList())
    def test_rotations(self, element):
        par_ref = self.s.copyObj(element)
        par_ref["name"] = par_ref["name"] + "_ref"
        self.funcSelect[self.s.system[element["name"]]["type"]](par_ref)

        name_test = element["name"]
        name_ref = par_ref["name"]

        rotation0 = np.array([12, -42, 187])
        rotation0 = np.array([-97.3, 237, 66])

        pivot0 = np.array([1, 5, 3])
        pivot0 = np.array([10, 50, 30])

        # rotate test parabola by given amount
        self.s.rotateGrids(name_test, rotation0, pivot=pivot0)
        
        rotation0abs = np.array([135, -67, 1.8])
        pivot0abs = np.array([1.8, 2, -43])
        
        self.s.rotateGrids(name_test, rotation0abs, pivot=pivot0abs, mode="absolute")
        self.s.rotateGrids(name_ref, rotation0abs, pivot=pivot0abs, mode="absolute")
        
        for r0ref, r0test in zip(self.s.system[name_ref]["ori"], self.s.system[name_test]["ori"]):
            self.assertAlmostEqual(r0ref, r0test)
   
        g0atest = self.s.generateGrids(name_test)
        g0aref = self.s.generateGrids(name_ref)

        for nx0atest, nx0aref in zip(g0atest.nx.ravel(), g0aref.nx.ravel()):
            self.assertAlmostEqual(nx0atest, nx0aref)
        
        for ny0atest, ny0aref in zip(g0atest.ny.ravel(), g0aref.ny.ravel()):
            self.assertAlmostEqual(ny0atest, ny0aref)
        
        for nz0atest, nz0aref in zip(g0atest.nz.ravel(), g0aref.nz.ravel()):
            self.assertAlmostEqual(nz0atest, nz0aref)
        
        for a0atest, a0aref in zip(g0atest.area.ravel(), g0aref.area.ravel()):
            self.assertAlmostEqual(a0atest, a0aref)
    
if __name__ == "__main__":
    import nose2
    nose2.main()
