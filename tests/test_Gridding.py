import unittest
import numpy as np
import random
from src.PyPO.System import System

class Test_Gridding(unittest.TestCase):
    def test_gcenterParabola(self):
        for i in range(10):
            self._random_parabola()

            grids = self.s.generateGrids("parabola")
            x_c = np.sum(grids.x) / grids.x.size
            y_c = np.sum(grids.y) / grids.y.size

            self.assertAlmostEqual(x_c, self.s.system["parabola"]["gcenter"][0], delta=1e-3)
            self.assertAlmostEqual(y_c, self.s.system["parabola"]["gcenter"][1], delta=1e-3)
    
    def test_gcenterHyperbola(self):
        for i in range(10):
            self._random_hyperbola()

            grids = self.s.generateGrids("hyperbola")
            x_c = np.sum(grids.x) / grids.x.size
            y_c = np.sum(grids.y) / grids.y.size

            self.assertAlmostEqual(x_c, self.s.system["hyperbola"]["gcenter"][0], delta=1e-3)
            self.assertAlmostEqual(y_c, self.s.system["hyperbola"]["gcenter"][1], delta=1e-3)
    
    def test_gcenterEllipse_z(self):
        for i in range(10):
            self._random_ellipse()

            grids = self.s.generateGrids("ellipse")
            x_c = np.sum(grids.x) / grids.x.size
            y_c = np.sum(grids.y) / grids.y.size

            self.assertAlmostEqual(x_c, self.s.system["ellipse"]["gcenter"][0], delta=1e-3)
            self.assertAlmostEqual(y_c, self.s.system["ellipse"]["gcenter"][1], delta=1e-3)
    
    def test_gcenterEllipse_x(self):
        for i in range(10):
            self._random_ellipse(ori="x")
            grids = self.s.generateGrids("ellipse", transform=False)
            x_c = np.sum(grids.x) / grids.x.size
            y_c = np.sum(grids.y) / grids.y.size
            
            self.assertAlmostEqual(x_c, self.s.system["ellipse"]["gcenter"][0], delta=1e-3)
            self.assertAlmostEqual(y_c, self.s.system["ellipse"]["gcenter"][1], delta=1e-3)
    
    def test_gcenterPlane(self):
        for i in range(10):
            self._random_plane()

            grids = self.s.generateGrids("plane")
            x_c = np.sum(grids.x) / grids.x.size
            y_c = np.sum(grids.y) / grids.y.size

            self.assertAlmostEqual(x_c, self.s.system["plane"]["gcenter"][0], delta=1e-3)
            self.assertAlmostEqual(y_c, self.s.system["plane"]["gcenter"][1], delta=1e-3)

    def _random_parabola(self):
        self.s = System(verbose=False)
        
        focus = np.array([0, 0, random.uniform(0,10)])

        gcenter = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

        parabola = {
                "name"      : "parabola",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : focus,
                "vertex"    : np.zeros(3),
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gcenter"   : gcenter,
                "gridsize"  : np.array([1001, 1001])
                }

        self.s.addParabola(parabola)
    
    def _random_hyperbola(self):
        self.s = System(verbose=False)
        
        focus1 = np.array([0, 0, random.uniform(0,10)])
        focus2 = np.array([0, 0, random.uniform(0,-10)])

        gcenter = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])
        ecc = random.uniform(1.01, 1.1)

        hyperbola  = {
                "name"      : "hyperbola",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : focus1,
                "focus_2"   : focus2,
                "ecc"       : ecc,
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gcenter"   : gcenter,
                "gridsize"  : np.array([1001, 1001])
                }

        self.s.addHyperbola(hyperbola)

    def _random_ellipse(self, ori="z"):
        self.s = System(verbose=False)
        
        if ori == "z":
            focus1 = np.array([0, 0, random.uniform(9,10)])
            focus2 = np.array([0, 0, random.uniform(-9,-10)])
        
        if ori == "x":
            focus1 = np.array([random.uniform(9,10), 0, 0])
            focus2 = np.array([random.uniform(-9,-10), 0, 0])

        gcenter = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        ecc = random.uniform(0.1, 0.9)

        ellipse  = {
                "name"      : "ellipse",
                "gmode"     : "uv",
                "pmode"     : "focus",
                "focus_1"   : focus1,
                "focus_2"   : focus2,
                "orient"    : ori,      
                "ecc"       : ecc,
                "lims_u"    : np.array([0, 0.1]),
                "lims_v"    : np.array([0, 360]),
                "gcenter"   : gcenter,
                "gridsize"  : np.array([1001, 1001])
                }

        self.s.addEllipse(ellipse)
    
    def _random_plane(self):
        self.s = System(verbose=False)

        gcenter = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

        plane  = {
                "name"      : "plane",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gcenter"   : gcenter,
                "gridsize"  : np.array([1001, 1001])
                }

        self.s.addPlane(plane)
if __name__ == "__main__":
    unittest.main()
