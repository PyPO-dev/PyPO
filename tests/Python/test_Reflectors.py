import sys
import matplotlib.pyplot as pt
sys.path.append('../../')

import unittest
import numpy as np
import scipy.interpolate as interp

import src.Python.Reflectors as Reflectors

class TestParabola(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        print("\nTesting parabolic reflector")
    
    def setUp(self):
        self.offTrans = np.array([0, 0, 0])
        self.rot = np.radians([0, 0, 0])
        self.cRot = np.array([0, 0, 0])
        
        self.a = 100
        self.b = 100
        
        self.lims_x = [-5000, 5000]
        self.lims_y = [-5000, 5000]
        
        self.lims_u = [200 / self.a, 5000/ self.a]
        self.lims_v = [0, 2*np.pi]
        
        self.gridsize = [101, 101]
        
        self.parabola = Reflectors.Parabola(a = self.a, b = self.b, cRot = self.cRot, offRot = self.rot, offTrans = self.offTrans, name = "p")
        
    def TearDown(self):
        pass
    
    def test_set_cRot(self):
        cRot_test = np.array([345, 129, 2343])
        
        self.parabola.set_cRot(cRot=cRot_test)
        
        for x, y in zip(cRot_test, self.parabola.cRot):
            self.assertEqual(x, y)
            
    def test_set_offTrans(self):
        offTrans_test = np.array([345, 129, 2343])
        
        self.parabola.set_offTrans(offTrans=offTrans_test)
        
        for x, y in zip(offTrans_test, self.parabola.offTrans):
            self.assertEqual(x, y)
            
    def test_set_offRot(self):
        offRot_test = np.array([47, 124, 36])
        
        self.parabola.set_offRot(offRot=offRot_test)
        
        for x, y in zip(offRot_test, self.parabola.offRot):
            self.assertEqual(x, y)
            
    def test_setGrid(self):
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, param='xy')
        
        self.assertEqual(self.parabola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.parabola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.parabola.setGrid(self.lims_u, self.lims_v, self.gridsize, param='uv')
        
        self.assertEqual(self.parabola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.parabola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
    
    def test_interpReflector(self):
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, param='xy')
        
        self.parabola.interpReflector(res=1)
        
        x = self.parabola.grid_x
        y = self.parabola.grid_y
        z = self.parabola.grid_z
        
        nx = self.parabola.grid_nx
        ny = self.parabola.grid_ny
        nz = self.parabola.grid_nz
        
        interp_z = interp.bisplev(x[:,0], y[0,:], self.parabola.tcks[0])
        
        interp_nx = interp.bisplev(x[:,0], y[0,:], self.parabola.tcks[1])
        interp_ny = interp.bisplev(x[:,0], y[0,:], self.parabola.tcks[2])
        interp_nz = interp.bisplev(x[:,0], y[0,:], self.parabola.tcks[3])
        
        norm = np.sqrt(interp_nz**2 + interp_ny**2 + interp_nx**2)
        
        interp_nx /= norm
        interp_ny /= norm
        interp_nz /= norm
        
        for i in range(len(x.ravel())):
            self.assertAlmostEqual(z.ravel()[i], interp_z.ravel()[i], places=3)
      
            self.assertAlmostEqual(nx.ravel()[i], interp_nx.ravel()[i], places=3)
            self.assertAlmostEqual(ny.ravel()[i], interp_ny.ravel()[i], places=3)
            self.assertAlmostEqual(nz.ravel()[i], interp_nz.ravel()[i], places=3)
            
    def test_uvParabola(self):
        u = np.array([100]).astype(float)
        v = 1/4 * np.pi
        
        x, y, z, nx, ny, nz = self.parabola.uvParabola(u, v)
        
        self.assertEqual(type(u), type(x))
        self.assertEqual(type(u), type(y))
        self.assertEqual(type(u), type(z))
        
        self.assertEqual(type(u), type(nx))
        self.assertEqual(type(u), type(ny))
        self.assertEqual(type(u), type(nz))
        
    def test_xyParabola(self):
        x = np.array([100]).astype(float)
        y = np.array([1/4 * np.pi]).astype(float)
        
        x, y, z, nx, ny, nz = self.parabola.xyParabola(x, y)
        
        self.assertEqual(type(x), type(x))
        self.assertEqual(type(x), type(y))
        self.assertEqual(type(x), type(z))
        
        self.assertEqual(type(x), type(nx))
        self.assertEqual(type(x), type(ny))
        self.assertEqual(type(x), type(nz))

class TestHyperbola(unittest.TestCase): 
    
    @classmethod
    def setUpClass(cls):
        print("\nTesting hyperbolic reflector")
    
    def setUp(self):
        self.offTrans = np.array([0, 0, 0])
        self.rot = np.radians([0, 0, 0])
        self.cRot = np.array([0, 0, 0])
        
        self.lims_x = [-310, 310]
        self.lims_y = [-310, 310]
        
        self.lims_u = [1, 100]
        self.lims_v = [0, 2*np.pi]
        
        self.gridsize = [101, 101]
        
        self.hyperbola = Reflectors.Hyperbola(a = 2590.5, b = 2590.5, c = 5606 / 2, cRot = self.cRot, offRot = self.rot, offTrans = self.offTrans, name = "h")
        
    def TearDown(self):
        pass
    
    def test_set_cRot(self):
        cRot_test = np.array([345, 129, 2343])
        
        self.hyperbola.set_cRot(cRot=cRot_test)
        
        for x, y in zip(cRot_test, self.hyperbola.cRot):
            self.assertEqual(x, y)
            
    def test_set_offTrans(self):
        offTrans_test = np.array([345, 129, 2343])
        
        self.hyperbola.set_offTrans(offTrans=offTrans_test)
        
        for x, y in zip(offTrans_test, self.hyperbola.offTrans):
            self.assertEqual(x, y)
            
    def test_set_offRot(self):
        offRot_test = np.array([47, 124, 36])
        
        self.hyperbola.set_offRot(offRot=offRot_test)
        
        for x, y in zip(offRot_test, self.hyperbola.offRot):
            self.assertEqual(x, y)
    
    def test_setGrid(self):
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, param='xy')
        
        self.assertEqual(self.hyperbola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.hyperbola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.hyperbola.setGrid(self.lims_u, self.lims_v, self.gridsize, param='uv')
        
        self.assertEqual(self.hyperbola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.hyperbola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
    
    def test_interpReflector(self):
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, calcArea=False, trunc=False, param='xy')
        
        self.hyperbola.interpReflector(res=1)
        
        x = self.hyperbola.grid_x
        y = self.hyperbola.grid_y
        z = self.hyperbola.grid_z
        
        nx = self.hyperbola.grid_nx
        ny = self.hyperbola.grid_ny
        nz = self.hyperbola.grid_nz
        
        interp_z = interp.bisplev(x[:,0], y[0,:], self.hyperbola.tcks[0])
        
        interp_nx = interp.bisplev(x[:,0], y[0,:], self.hyperbola.tcks[1])
        interp_ny = interp.bisplev(x[:,0], y[0,:], self.hyperbola.tcks[2])
        interp_nz = interp.bisplev(x[:,0], y[0,:], self.hyperbola.tcks[3])
        
        norm = np.sqrt(interp_nz**2 + interp_ny**2 + interp_nx**2)
        
        interp_nx /= norm
        interp_ny /= norm
        interp_nz /= norm
        
        for i in range(len(x.ravel())):
            self.assertAlmostEqual(z.ravel()[i], interp_z.ravel()[i], places=3)

            self.assertAlmostEqual(nx.ravel()[i], interp_nx.ravel()[i], places=3)
            self.assertAlmostEqual(ny.ravel()[i], interp_ny.ravel()[i], places=3)
            self.assertAlmostEqual(nz.ravel()[i], interp_nz.ravel()[i], places=3)
    
    def test_uvHyperbola(self):
        u = np.array([100]).astype(float)
        v = 1/4 * np.pi
        
        x, y, z, nx, ny, nz = self.hyperbola.uvHyperbola(u, v)
        
        self.assertEqual(type(u), type(x))
        self.assertEqual(type(u), type(y))
        self.assertEqual(type(u), type(z))
        
        self.assertEqual(type(u), type(nx))
        self.assertEqual(type(u), type(ny))
        self.assertEqual(type(u), type(nz))
        
    def test_xyHyperbola(self):
        x = np.array([100]).astype(float)
        y = np.array([1/4 * np.pi]).astype(float)
        
        x, y, z, nx, ny, nz = self.hyperbola.xyHyperbola(x, y)
        
        self.assertEqual(type(x), type(x))
        self.assertEqual(type(x), type(y))
        self.assertEqual(type(x), type(z))
        
        self.assertEqual(type(x), type(nx))
        self.assertEqual(type(x), type(ny))
        self.assertEqual(type(x), type(nz))
    
if __name__ == "__main__":
    unittest.main()
        
