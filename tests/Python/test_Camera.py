import sys

sys.path.append('../../')

import unittest
import numpy as np
import scipy.interpolate as interp

import src.POPPy.MatRotate as MatRotate

import src.POPPy.Camera as Camera
import src.POPPy.Copy as Copy

class TestCamera(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\nTesting Camera")
    
    def setUp(self):
        
        self.center = np.array([0,0,3500])
        self.lims_x = [-100, 100]
        self.lims_y = [-100, 100]
        self.gridsize = [501, 501]
        
        self.name = "cam_test"
        
        self.offRot = np.radians([0, 0, 0])
        
        self.camera = Camera.Camera(center=self.center, name=self.name, units='mm')
        
    def test_setGrid(self):
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        
        self.assertEqual(self.camera.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.camera.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.camera.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
    def test_translateGrid(self):
        offTrans = np.array([3, 1, -5])
        
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        
        x0 = Copy.copyGrid(self.camera.grid_x.ravel())
        y0 = Copy.copyGrid(self.camera.grid_y.ravel())
        z0 = Copy.copyGrid(self.camera.grid_z.ravel())
        
        c0 = Copy.copyGrid(self.camera.center)
        
        self.camera.translateGrid(offTrans)
        
        x = self.camera.grid_x.ravel()
        y = self.camera.grid_y.ravel()
        z = self.camera.grid_z.ravel()
        
        for (xx0, yy0, zz0, xx, yy, zz) in zip(x0, y0, z0, x, y, z):
            self.assertEqual(xx0 + offTrans[0], xx)
            self.assertEqual(yy0 + offTrans[1], yy)
            self.assertEqual(zz0 + offTrans[2], zz)
            
        for cc0, cc, tr in zip(c0, self.camera.center, offTrans):
            self.assertEqual(cc0 + tr, cc)
        
    def test_rotateGrid(self):
        offRot = np.array([34, 21, 178])
        
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        
        x0 = Copy.copyGrid(self.camera.grid_x.ravel())
        y0 = Copy.copyGrid(self.camera.grid_y.ravel())
        z0 = Copy.copyGrid(self.camera.grid_z.ravel())
        
        nx0 = Copy.copyGrid(self.camera.grid_nx.ravel())
        ny0 = Copy.copyGrid(self.camera.grid_ny.ravel())
        nz0 = Copy.copyGrid(self.camera.grid_nz.ravel())
        
        c0 = Copy.copyGrid(self.camera.center)
        
        self.camera.rotateGrid(offRot)
        
        x = self.camera.grid_x.ravel()
        y = self.camera.grid_y.ravel()
        z = self.camera.grid_z.ravel()
        
        nx = self.camera.grid_nx.ravel()
        ny = self.camera.grid_ny.ravel()
        nz = self.camera.grid_nz.ravel()
        
        for xx0, yy0, zz0, xx, yy, zz, nxx0, nyy0, nzz0, nxx, nyy, nzz in zip(x0, y0, z0, x, y, z, nx0, ny0, nz0, nx, ny, nz):
            pos_t = MatRotate.MatRotate(offRot, [xx0, yy0, zz0], origin=self.center)
            norm_t = MatRotate.MatRotate(offRot, [nxx0, nyy0, nzz0], vecRot=True)
            
            self.assertEqual(pos_t[0], xx)
            self.assertEqual(pos_t[1], yy)
            self.assertEqual(pos_t[2], zz)
            
            self.assertEqual(norm_t[0], nxx)
            self.assertEqual(norm_t[1], nyy)
            self.assertEqual(norm_t[2], nzz)
            
            
        c0_t = MatRotate.MatRotate(offRot, c0, origin=self.center)
        for cc0, cc in zip(c0, self.camera.center):
            self.assertEqual(cc0, cc)
        
    def test_interpCamera(self):
        self.camera.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy')
        
        self.camera.interpCamera(res=1, mode='z')
        
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
            
    def test_get_conv(self):
        check1 = self.camera.get_conv('mm')
        check2 = self.camera.get_conv('cm')
        check3 = self.camera.get_conv('m')
        
        check4 = self.camera.get_conv('deg')
        check5 = self.camera.get_conv('am')
        check6 = self.camera.get_conv('as')
        
        self.assertEqual(check1, 1)
        self.assertEqual(check2, 1e2)
        self.assertEqual(check3, 1e3)
        
        self.assertEqual(check4, np.pi / 180)
        self.assertEqual(check5, np.pi / (180 * 60))
        self.assertEqual(check6, np.pi / (180 * 3600))
        
if __name__ == "__main__":
    unittest.main()
 
