import sys

sys.path.append('../../')

import unittest
import numpy as np
import scipy.interpolate as interp

import src.Python.Camera as Camera

class TestMatRotate(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\nTesting Camera")
    
    def setUp(self):
        
        self.center = np.array([0,0,3500])
        self.lims_x = [-100, 100]
        self.lims_y = [-100, 100]
        self.gridsize = [501, 501]
        
        self.name = "cam_test"
        
        self.offTrans = np.array([0, 0, 0])
        self.offRot = np.radians([0, 0, 0])
        
        self.camera = Camera.Camera(center=self.center, offTrans=self.offTrans, offRot=self.offRot, name=self.name)
        
    def test_setGrid(self):
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize)
        
        self.assertEqual(self.camera.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.camera.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
    def test_interpCamera(self):
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize)
        
        self.camera.interpCamera(res=1)
        
        x = self.camera.grid_x
        y = self.camera.grid_y
        z = self.camera.grid_z
        
        nx = self.camera.grid_nx
        ny = self.camera.grid_ny
        nz = self.camera.grid_nz
        
        interp_z = interp.bisplev(x[:,0], y[0,:], self.camera.tcks[0])
        
        interp_nx = interp.bisplev(x[:,0], y[0,:], self.camera.tcks[1])
        interp_ny = interp.bisplev(x[:,0], y[0,:], self.camera.tcks[2])
        interp_nz = interp.bisplev(x[:,0], y[0,:], self.camera.tcks[3])
        
        norm = np.sqrt(interp_nz**2 + interp_ny**2 + interp_nx**2)
        
        interp_nx /= norm
        interp_ny /= norm
        interp_nz /= norm
        
        for i in range(len(x.ravel())):
            self.assertAlmostEqual(z.ravel()[i], interp_z.ravel()[i], places=7)
      
            self.assertAlmostEqual(nx.ravel()[i], interp_nx.ravel()[i], places=7)
            self.assertAlmostEqual(ny.ravel()[i], interp_ny.ravel()[i], places=7)
            self.assertAlmostEqual(nz.ravel()[i], interp_nz.ravel()[i], places=7)
        
if __name__ == "__main__":
    unittest.main()
 
