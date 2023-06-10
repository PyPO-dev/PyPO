import sys
import random

#sys.path.append('../../')

import unittest
import numpy as np

from PyPO.System import System
##
# @file
# 
# Script for testing the transformational formalism of PyPO.

class Test_MatTransform(unittest.TestCase):
    def test_translations_parabola(self):
        self._run_translations("parabola_test", "parabola_ref")
    
    def test_translations_hyperbola(self):
        self._run_translations("hyperbola_test", "hyperbola_ref")
    
    def test_translations_ellipse(self):
        self._run_translations("ellipse_test", "ellipse_ref")
    
    def test_translations_plane(self):
        self._run_translations("plane_test", "plane_ref")

    def test_rotations_parabola(self):
        self._run_rotations("parabola_test", "parabola_ref")
    
    def test_rotations_hyperbola(self):
        self._run_rotations("hyperbola_test", "hyperbola_ref")
    
    def test_rotations_ellipse(self):
        self._run_rotations("ellipse_test", "ellipse_ref")
    
    def test_rotations_plane(self):
        self._run_rotations("plane_test", "plane_ref")

    def _run_rotations(self, name_test, name_ref):
        for i in range(10):
            self._make_random_system()
            
            rotation0 = np.degrees((np.random.rand(3) - 0.5) * 2*np.pi)
            rotation1 = np.degrees((np.random.rand(3) - 0.5) * 2*np.pi)

            pivot0 = np.random.rand(3)
            pivot1 = np.random.rand(3)


            # rotate test parabola by given amount
            self.s.rotateGrids(name_test, rotation0, pivot=pivot0)
            
            rotation0abs = np.degrees((np.random.rand(3) - 0.5) * 2*np.pi)
            pivot0abs = np.random.rand(3)
            
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
        
            #self.s.homeReflector(name_test)
            #self.s.homeReflector(name_ref)
            del self.s
    
    def _run_translations(self, name_test, name_ref):
        for i in range(10):
            self._make_random_system()
            
            translation0 = np.random.rand(3)
            translation1 = np.random.rand(3)

            pos_add0 = self.s.copyObj(self.s.system[name_test]["pos"])
            # translate test parabola by given amount
            self.s.translateGrids(name_test, translation0)
        
            for x0, x0test in zip(translation0 + pos_add0, self.s.system[name_test]["pos"]):
                self.assertAlmostEqual(x0, x0test, delta=1e-6)
            
            # translate test parabola again
            self.s.translateGrids(name_test, translation1)
        
            for x1, x1test in zip(translation1 + translation0 + pos_add0, self.s.system[name_test]["pos"]):
                self.assertAlmostEqual(x1, x1test, delta=1e-6)

            # Test absolute translations
            translation0abs = np.random.rand(3)

            self.s.translateGrids(name_test, translation0abs, mode="absolute")
            self.s.translateGrids(name_ref, translation0abs, mode="absolute")

            # Test if pos is translated properly
            for x0a, x0atest in zip(translation0abs, self.s.system[name_test]["pos"]):
                self.assertAlmostEqual(x0a, x0atest, delta=1e-6)

            # Unpack grids and check if equal
            g0atest = self.s.generateGrids(name_test)
            g0aref = self.s.generateGrids(name_ref)

            for x0atest, x0aref in zip(g0atest.x.ravel(), g0aref.x.ravel()):
                self.assertAlmostEqual(x0atest, x0aref, delta=1e-6)
            
            for y0atest, y0aref in zip(g0atest.y.ravel(), g0aref.y.ravel()):
                self.assertAlmostEqual(y0atest, y0aref, delta=1e-6)
            
            for z0atest, z0aref in zip(g0atest.z.ravel(), g0aref.z.ravel()):
                self.assertAlmostEqual(z0atest, z0aref, delta=1e-6)

            del self.s

    def _make_random_system(self):
        self.s = System(verbose=False)

        rand1 = np.random.rand(3)
        rand2 = np.random.rand(3)
        
        rand7 = np.random.rand(1)
        
        parabola_ref = {
                "name"      : "parabola_ref",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : rand1,
                "vertex"    : rand2,
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        self.s.addParabola(parabola_ref)
   
        parabola_test = self.s.copyObj(parabola_ref)
        parabola_test["name"] = "parabola_test"

        self.s.addParabola(parabola_test)
        
        hyperbola_ref = {
                "name"      : "hyperbola_ref",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : rand1,
                "focus_2"   : rand2,
                "ecc"       : rand7[0] + 1,
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        self.s.addHyperbola(hyperbola_ref)
   
        hyperbola_test = self.s.copyObj(hyperbola_ref)
        hyperbola_test["name"] = "hyperbola_test"

        self.s.addHyperbola(hyperbola_test)
        
        ellipse_ref = {
                "name"      : "ellipse_ref",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : rand1,
                "focus_2"   : rand2,
                "ecc"       : rand7[0],
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        self.s.addEllipse(ellipse_ref)
   
        ellipse_test = self.s.copyObj(ellipse_ref)
        ellipse_test["name"] = "ellipse_test"

        self.s.addEllipse(ellipse_test)

        plane_ref = {
                "name"      : "plane_ref",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        self.s.addPlane(plane_ref)
   
        plane_test = self.s.copyObj(plane_ref)
        plane_test["name"] = "plane_test"

        self.s.addPlane(plane_test)

    #def tearDown(self):
    #    del self.s

if __name__ == "__main__":
    unittest.main()
