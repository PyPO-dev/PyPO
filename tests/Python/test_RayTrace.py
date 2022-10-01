import sys

sys.path.append('../../')

import unittest
import numpy as np

from src.POPPy.RayTrace import RayTrace
from src.POPPy.Camera import Camera
from src.POPPy.MatRotate import MatRotate

class TestRayTrace(unittest.TestCase):
    """
    Test class for raytrace object.
    """
    
    @classmethod
    def setUpClass(cls):
        print("\nTesting RayTrace")
    
    def setUp(self):
        self.nRays = 0
        self.nRing = 0
        self.a = 0
        self.b = 0
        self.angx = 0
        self.angy = 0
        self.originChief = np.zeros(3)
        self.tiltChief = np.zeros(3)
        
        self.RT = RayTrace()
        
        self.RT.initRaytracer(self.nRays,
                            self.nRing,
                            self.a,
                            self.b,
                            self.angx,
                            self.angy,
                            self.originChief,
                            self.tiltChief)
        
        self.centercam = np.array([0,0,0])
        self.lims_x = [-100, 100]
        self.lims_y = [-100, 100]
        self.gridsize = [101, 101]
        
        self.name = "cam_test"
        self.offRot = np.radians([0, 0, 0])
        
        self.camera = Camera(center=self.centercam, name=self.name, units='mm')
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        self.camera.interpCamera(res=1, mode='z')
        self.tcks = self.camera.tcks
    
    def test_set_tcks(self):
        
        self.RT.set_tcks(self.tcks)
        
        for i, t in enumerate(self.tcks):
            self.assertEqual(self.RT.tcks[i], t)
            
    def test_getMode(self):
        # First check up along z
        self.assertEqual('z', self.RT.getMode())
        
        # Now tilt 44 degrees along x and y
        tiltx = np.array([44,0,0])
        tilty = np.array([0,44,0])
        
        RTtest = RayTrace()
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tiltx)
        
        self.assertEqual('z', RTtest.getMode())
        
        RTtest.clearRays()
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tilty)
        
        self.assertEqual('z', RTtest.getMode())
        
        RTtest.clearRays()
        
        # Now tilt 46 degrees along x and y
        tiltx = np.array([46,0,0])
        tilty = np.array([0,46,0])
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tiltx)
        
        self.assertEqual('y', RTtest.getMode())
        
        RTtest.clearRays()
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tilty)
        
        self.assertEqual('x', RTtest.getMode())
        
        RTtest.clearRays()
        
        # Now tilt 45 degrees along x and y. Should resolve to x and y.
        tiltx = np.array([45,0,0])
        tilty = np.array([0,45,0])
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tiltx)
        
        self.assertEqual('z', RTtest.getMode())
        
        RTtest.clearRays()
        
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tilty)
        
        self.assertEqual('z', RTtest.getMode())
        
        RTtest.clearRays()

    def test_interpEval(self):
        self.RT.set_tcks(self.tcks)
        
        x1 = np.linspace(-100, 100)
        x2 = np.linspace(-100, 100)
        
        for xx1, xx2 in zip(x1, x2):
            self.assertEqual(self.RT._interpEval(xx1, xx2), 0)
            
    def test_interpEval_n(self):
        self.RT.set_tcks(self.tcks)
        
        x1 = np.linspace(-100, 100)
        x2 = np.linspace(-100, 100)
        
        for xx1, xx2 in zip(x1, x2):
            interp_n = self.RT._interpEval_n(xx1, xx2)
            self.assertEqual(interp_n[0], 0)
            self.assertEqual(interp_n[1], 0)
            self.assertEqual(interp_n[2], 1)
            
    def test_optLine(self):
        off_z = 100
        ray_test = self.RT.rays["ray_0"]
        a_test = 50
        
        x1 = 0
        x2 = 1
        x3 = 2
        
        self.camera.translateGrid(np.array([0,0,off_z]))
        self.camera.interpCamera(res=1, mode='z')
        self.RT.set_tcks(self.camera.tcks)
        
        diff = self.RT._optLine(a_test, ray_test, x1, x2, x3)
        
        self.assertAlmostEqual(a_test, diff)
        
    def test_propagateRays(self):
        off_z = 100
        

        tiltx = np.array([25,0,0])
        dy = -np.tan(np.radians(tiltx[0])) * off_z
        
        RTtest = RayTrace()
        
        # TEST 1
        RTtest.initRaytracer(self.nRays,
                    self.nRing,
                    self.a,
                    self.b,
                    self.angx,
                    self.angy,
                    self.originChief,
                    tiltChief=tiltx)
        
        mode = RTtest.getMode()
        
        self.camera.translateGrid(np.array([0,0,off_z]))
        self.camera.interpCamera(res=1, mode=mode)
        RTtest.set_tcks(self.camera.tcks)
        
        RTtest.propagateRays(a0=1, mode=mode)
        
        self.assertAlmostEqual(RTtest.rays["ray_0"]["positions"][0][1] + dy, RTtest.rays["ray_0"]["positions"][1][1])
        
        RTtest.clearRays()
        
        # TEST 2
        RTtest.initRaytracer(nRays=4,
                    nRing=0,
                    a=0,
                    b=0,
                    angx=6,
                    angy=6,
                    originChief=np.zeros(3),
                    tiltChief=np.zeros(3))
        
        RTtest.set_tcks(self.camera.tcks)
        
        dr = abs(np.tan(np.radians(6) * off_z))
        
        RTtest.propagateRays(a0=1, mode=mode)
        
        for i, (key, ray) in enumerate(RTtest.rays.items()):
            diff = np.array([ray["positions"][1][0] - ray["positions"][0][0], ray["positions"][1][1] - ray["positions"][0][1]])
        
            diff = np.sqrt(np.dot(diff, diff))
            
            if i == 0:
                self.assertAlmostEqual(diff, 0)
                
            else:
                self.assertAlmostEqual(diff, dr)
        
        RTtest.clearRays()
        
        # TEST 3
        RTtest.initRaytracer(nRays=4,
                    nRing=0,
                    a=5,
                    b=5,
                    angx=6,
                    angy=6,
                    originChief=np.zeros(3),
                    tiltChief=np.zeros(3))
        
        RTtest.set_tcks(self.camera.tcks)
        
        dr = abs(np.tan(np.radians(6) * off_z))
        
        RTtest.propagateRays(a0=1, mode=mode)
        
        for i, (key, ray) in enumerate(RTtest.rays.items()):
            diff = np.array([ray["positions"][1][0] - ray["positions"][0][0], ray["positions"][1][1] - ray["positions"][0][1]])
        
            diff = np.sqrt(np.dot(diff, diff))
            
            if i == 0:
                self.assertAlmostEqual(diff, 0)
                
            else:
                self.assertAlmostEqual(diff, dr+5)
        
        RTtest.clearRays()
        
        # TEST 4
        RTtest.initRaytracer(nRays=4,
                    nRing=0,
                    a=0,
                    b=0,
                    angx=6,
                    angy=6,
                    originChief=np.zeros(3),
                    tiltChief=tiltx)
        
        RTtest.set_tcks(self.camera.tcks)
        
        dr = abs(np.tan(np.radians(6) * off_z))
        
        RTtest.propagateRays(a0=1, mode=mode)
        
        for i, (key, ray) in enumerate(RTtest.rays.items()):
            diff = np.array([ray["positions"][1][0] - ray["positions"][0][0], ray["positions"][1][1] - (ray["positions"][0][1] + dy)])
        
            diff = np.sqrt(np.dot(diff, diff))
            
            if i == 0:
                self.assertAlmostEqual(diff, 0)
                
            else:
                self.assertAlmostEqual(diff, dr)
        
        RTtest.clearRays()
        
        # TEST 5
        RTtest.initRaytracer(nRays=4,
                    nRing=4,
                    a=5,
                    b=5,
                    angx=0,
                    angy=0,
                    originChief=np.zeros(3),
                    tiltChief=np.zeros(3))

        RTtest.set_tcks(self.camera.tcks)
        RTtest.propagateRays(a0=1, mode=mode)
        
        
        camera2 = Camera(center=np.zeros(3), name="cam2", units='mm')
        camera2.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        camera2.interpCamera(res=1, mode='z')

        RTtest.set_tcks(camera2.tcks)
        RTtest.propagateRays(a0=1, mode=mode)

        for i, (key, ray) in enumerate(RTtest.rays.items()):
            for x1, x2 in zip(ray["positions"][0], ray["positions"][-1]):
                self.assertAlmostEqual(x1, x2)
                
            for x1, x2 in zip(ray["directions"][0], ray["directions"][-1]):
                self.assertAlmostEqual(x1, x2)

    def tearDown(self):
        pass
    
    def test_set_tcks(self):
        pass

if __name__ == "__main__":
    unittest.main()
 
