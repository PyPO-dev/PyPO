import sys
import matplotlib.pyplot as pt
sys.path.append('../../')

import unittest
import numpy as np
import scipy.interpolate as interp

import src.POPPy.Copy as Copy
import src.POPPy.MatRotate as MatRotate
import src.POPPy.Reflectors as Reflectors

class TestParabola(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        print("\nTesting parabolic reflector")
    
    def setUp(self):
        self.offTrans = np.array([0, 0, 0])
        self.rot = np.radians([0, 0, 0])
        self.cRot = np.array([0, 0, 0])
        
        self.a = 10
        self.b = 10
        
        self.lims_x = [-500, 500]
        self.lims_y = [-500, 500]
        
        self.lims_u = [0, 500/ self.a]
        self.lims_v = [0, 2*np.pi]
        
        self.gridsize = [201, 201]
        
        self.parabola = Reflectors.Parabola(a = self.a, b = self.b, cRot = self.cRot, name = "p", units='mm')
        
    def TearDown(self):
        pass

    def test_set_cRot(self):
        cRot_test = np.array([345, 129, 2343])
        
        self.parabola.set_cRot(cRot=cRot_test)
        
        for x, y in zip(cRot_test, self.parabola.cRot):
            self.assertEqual(x, y)
    
    def test_setGrid(self):
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        self.assertEqual(self.parabola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.parabola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.parabola.setGrid(self.lims_u, self.lims_v, self.gridsize, gmode='uv', axis='a', trunc=False, flip=False)
        
        self.assertEqual(self.parabola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.parabola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.parabola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
    def test_translateGrid(self):
        offTrans = np.array([3, 1, -5])
        
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        x0 = Copy.copyGrid(self.parabola.grid_x.ravel())
        y0 = Copy.copyGrid(self.parabola.grid_y.ravel())
        z0 = Copy.copyGrid(self.parabola.grid_z.ravel())

        self.parabola.translateGrid(offTrans)
        
        x = self.parabola.grid_x.ravel()
        y = self.parabola.grid_y.ravel()
        z = self.parabola.grid_z.ravel()
        
        for (xx0, yy0, zz0, xx, yy, zz) in zip(x0, y0, z0, x, y, z):
            self.assertEqual(xx0 + offTrans[0], xx)
            self.assertEqual(yy0 + offTrans[1], yy)
            self.assertEqual(zz0 + offTrans[2], zz)

    def test_rotateGrid(self):
        offRot = np.array([34, 21, 178])
        
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        x0 = Copy.copyGrid(self.parabola.grid_x.ravel())
        y0 = Copy.copyGrid(self.parabola.grid_y.ravel())
        z0 = Copy.copyGrid(self.parabola.grid_z.ravel())
        
        nx0 = Copy.copyGrid(self.parabola.grid_nx.ravel())
        ny0 = Copy.copyGrid(self.parabola.grid_ny.ravel())
        nz0 = Copy.copyGrid(self.parabola.grid_nz.ravel())

        self.parabola.rotateGrid(offRot)
        
        x = self.parabola.grid_x.ravel()
        y = self.parabola.grid_y.ravel()
        z = self.parabola.grid_z.ravel()
        
        nx = self.parabola.grid_nx.ravel()
        ny = self.parabola.grid_ny.ravel()
        nz = self.parabola.grid_nz.ravel()
        
        for xx0, yy0, zz0, xx, yy, zz, nxx0, nyy0, nzz0, nxx, nyy, nzz in zip(x0, y0, z0, x, y, z, nx0, ny0, nz0, nx, ny, nz):
            pos_t = MatRotate.MatRotate(offRot, [xx0, yy0, zz0], origin=self.cRot)
            norm_t = MatRotate.MatRotate(offRot, [nxx0, nyy0, nzz0], vecRot=True)
            
            self.assertEqual(pos_t[0], xx)
            self.assertEqual(pos_t[1], yy)
            self.assertEqual(pos_t[2], zz)
            
            self.assertEqual(norm_t[0], nxx)
            self.assertEqual(norm_t[1], nyy)
            self.assertEqual(norm_t[2], nzz)

    def test_interpReflector(self):
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        self.parabola.interpReflector(res=1, mode='z')
        
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

    def test_uvGrid(self):
        u = np.array([100]).astype(float)
        v = 1/4 * np.pi
        du = 1
        dv = 1
        
        x, y, z, nx, ny, nz, area = self.parabola.uvGrid(u, v, du=1, dv=1)
        
        self.assertEqual(type(u), type(x))
        self.assertEqual(type(u), type(y))
        self.assertEqual(type(u), type(z))
        
        self.assertEqual(type(u), type(nx))
        self.assertEqual(type(u), type(ny))
        self.assertEqual(type(u), type(nz))
        
    def test_xyGrid(self):
        x = np.array([100]).astype(float)
        y = np.array([1/4 * np.pi]).astype(float)
        
        x, y, z, nx, ny, nz = self.parabola.xyGrid(x, y)
        
        self.assertEqual(type(x), type(x))
        self.assertEqual(type(x), type(y))
        self.assertEqual(type(x), type(z))
        
        self.assertEqual(type(x), type(nx))
        self.assertEqual(type(x), type(ny))
        self.assertEqual(type(x), type(nz))
        
    def test_get_conv(self):
        check1 = self.parabola._get_conv('mm')
        check2 = self.parabola._get_conv('cm')
        check3 = self.parabola._get_conv('m')
        
        self.assertEqual(check1, 1)
        self.assertEqual(check2, 1e2)
        self.assertEqual(check3, 1e3)
    

    def test_homeReflector(self):
        # Apply three random rotations and translations and see if homing works
        self.parabola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        trans1 = np.random.rand(3)
        trans2 = np.random.rand(3)
        trans3 = np.random.rand(3)
        
        rot1 = np.random.rand(3) * 100
        rot2 = np.random.rand(3) * 100
        rot3 = np.random.rand(3) * 100
        
        cRot1 = np.random.rand(3)
        cRot2 = np.random.rand(3)
        cRot3 = np.random.rand(3)
        
        # Store otiginal grids
        x = Copy.copyGrid(self.parabola.grid_x)
        y = Copy.copyGrid(self.parabola.grid_y)
        z = Copy.copyGrid(self.parabola.grid_z)
        
        nx = Copy.copyGrid(self.parabola.grid_nx)
        ny = Copy.copyGrid(self.parabola.grid_ny)
        nz = Copy.copyGrid(self.parabola.grid_nz)
        
        self.parabola.translateGrid(trans1, units='m')
        self.parabola.set_cRot(cRot1, units='m')
        self.parabola.rotateGrid(rot1)
        self.parabola.translateGrid(trans2, units='m')
        self.parabola.set_cRot(cRot2, units='m')
        self.parabola.rotateGrid(rot2)
        self.parabola.translateGrid(trans3, units='m')
        self.parabola.set_cRot(cRot3, units='m')
        self.parabola.rotateGrid(rot3)

        self.parabola.homeReflector()

        for xp, xh, yp, yh, zp, zh, nxp, nxh, nyp, nyh, nzp, nzh in zip(x.ravel(), self.parabola.grid_x.ravel(),
                                                                        y.ravel(), self.parabola.grid_y.ravel(),
                                                                        z.ravel(), self.parabola.grid_z.ravel(),
                                                                        nx.ravel(), self.parabola.grid_nx.ravel(),
                                                                        ny.ravel(), self.parabola.grid_ny.ravel(),
                                                                        nz.ravel(), self.parabola.grid_nz.ravel()):
            self.assertAlmostEqual(xp, xh)
            self.assertAlmostEqual(yp, yh)
            self.assertAlmostEqual(zp, zh)
            
            self.assertAlmostEqual(nxp, nxh)
            self.assertAlmostEqual(nyp, nyh)
            self.assertAlmostEqual(nzp, nzh)

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
        
        self.lims_u = [1, 1.045]
        self.lims_v = [0, 2*np.pi]
        
        self.gridsize = [201, 201]
        self.sec = 'upper'
        
        self.hyperbola = Reflectors.Hyperbola(a = 1070, b = 1070, c = 2590, cRot = self.cRot, name = "h", sec = self.sec, units='mm')
        
    def TearDown(self):
        pass

    def test_set_cRot(self):
        cRot_test = np.array([345, 129, 2343])
        
        self.hyperbola.set_cRot(cRot=cRot_test)
        
        for x, y in zip(cRot_test, self.hyperbola.cRot):
            self.assertEqual(x, y)
            
    def test_setGrid(self):
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        self.assertEqual(self.hyperbola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.hyperbola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))
        
        # TEST: see if sec='upper' and sec='lower' are exactly mirrored
        hyperbola_test = Reflectors.Hyperbola(a = 1070, b = 1070, c = 2590, cRot = self.cRot, name = "ht", sec = 'lower', units='mm')
        hyperbola_test.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        diff = self.hyperbola.grid_z + hyperbola_test.grid_z

        for diff_el in diff.ravel():
            self.assertEqual(diff_el, 0)
        
        self.hyperbola.setGrid(self.lims_u, self.lims_v, self.gridsize, gmode='uv', axis='a', trunc=False, flip=False)
        
        self.assertEqual(self.hyperbola.grid_x.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_y.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_z.shape, (self.gridsize[0], self.gridsize[1]))
        
        self.assertEqual(self.hyperbola.grid_nx.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_ny.shape, (self.gridsize[0], self.gridsize[1]))
        self.assertEqual(self.hyperbola.grid_nz.shape, (self.gridsize[0], self.gridsize[1]))

    def test_translateGrid(self):
        offTrans = np.array([3, 1, -5])
        
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        x0 = Copy.copyGrid(self.hyperbola.grid_x.ravel())
        y0 = Copy.copyGrid(self.hyperbola.grid_y.ravel())
        z0 = Copy.copyGrid(self.hyperbola.grid_z.ravel())

        self.hyperbola.translateGrid(offTrans)
        
        x = self.hyperbola.grid_x.ravel()
        y = self.hyperbola.grid_y.ravel()
        z = self.hyperbola.grid_z.ravel()
        
        for (xx0, yy0, zz0, xx, yy, zz) in zip(x0, y0, z0, x, y, z):
            self.assertEqual(xx0 + offTrans[0], xx)
            self.assertEqual(yy0 + offTrans[1], yy)
            self.assertEqual(zz0 + offTrans[2], zz)
    
    def test_rotateGrid(self):
        offRot = np.array([34, 21, 178])
        
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        x0 = Copy.copyGrid(self.hyperbola.grid_x.ravel())
        y0 = Copy.copyGrid(self.hyperbola.grid_y.ravel())
        z0 = Copy.copyGrid(self.hyperbola.grid_z.ravel())
        
        nx0 = Copy.copyGrid(self.hyperbola.grid_nx.ravel())
        ny0 = Copy.copyGrid(self.hyperbola.grid_ny.ravel())
        nz0 = Copy.copyGrid(self.hyperbola.grid_nz.ravel())

        self.hyperbola.rotateGrid(offRot)
        
        x = self.hyperbola.grid_x.ravel()
        y = self.hyperbola.grid_y.ravel()
        z = self.hyperbola.grid_z.ravel()
        
        nx = self.hyperbola.grid_nx.ravel()
        ny = self.hyperbola.grid_ny.ravel()
        nz = self.hyperbola.grid_nz.ravel()
        
        for xx0, yy0, zz0, xx, yy, zz, nxx0, nyy0, nzz0, nxx, nyy, nzz in zip(x0, y0, z0, x, y, z, nx0, ny0, nz0, nx, ny, nz):
            pos_t = MatRotate.MatRotate(offRot, [xx0, yy0, zz0], origin=self.cRot)
            norm_t = MatRotate.MatRotate(offRot, [nxx0, nyy0, nzz0], vecRot=True)
            
            self.assertEqual(pos_t[0], xx)
            self.assertEqual(pos_t[1], yy)
            self.assertEqual(pos_t[2], zz)
            
            self.assertEqual(norm_t[0], nxx)
            self.assertEqual(norm_t[1], nyy)
            self.assertEqual(norm_t[2], nzz)
    
    def test_interpReflector(self):
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        self.hyperbola.interpReflector(res=1, mode='z')
        
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
            
    def test_uvGrid(self):
        u = np.array([100]).astype(float)
        v = 1/4 * np.pi
        
        x, y, z, nx, ny, nz, area = self.hyperbola.uvGrid(u, v, du=1, dv=1)
        
        self.assertEqual(type(u), type(x))
        self.assertEqual(type(u), type(y))
        self.assertEqual(type(u), type(z))
        
        self.assertEqual(type(u), type(nx))
        self.assertEqual(type(u), type(ny))
        self.assertEqual(type(u), type(nz))

    def test_xyGrid(self):
        x = np.array([100]).astype(float)
        y = np.array([1/4 * np.pi]).astype(float)
        
        x, y, z, nx, ny, nz = self.hyperbola.xyGrid(x, y)
        
        self.assertEqual(type(x), type(x))
        self.assertEqual(type(x), type(y))
        self.assertEqual(type(x), type(z))
        
        self.assertEqual(type(x), type(nx))
        self.assertEqual(type(x), type(ny))
        self.assertEqual(type(x), type(nz))
        
    def test_get_conv(self):
        check1 = self.hyperbola._get_conv('mm')
        check2 = self.hyperbola._get_conv('cm')
        check3 = self.hyperbola._get_conv('m')
        
        self.assertEqual(check1, 1)
        self.assertEqual(check2, 1e2)
        self.assertEqual(check3, 1e3)

    def test_homeReflector(self):
        # Apply three random rotations and translations and see if homing works
        self.hyperbola.setGrid(self.lims_x, self.lims_y, self.gridsize, gmode='xy', axis='a', trunc=False, flip=False)
        
        trans1 = np.random.rand(3)
        trans2 = np.random.rand(3)
        trans3 = np.random.rand(3)
        
        rot1 = np.random.rand(3) * 100
        rot2 = np.random.rand(3) * 100
        rot3 = np.random.rand(3) * 100
        
        cRot1 = np.random.rand(3)
        cRot2 = np.random.rand(3)
        cRot3 = np.random.rand(3)
        
        # Store original grids
        x = Copy.copyGrid(self.hyperbola.grid_x)
        y = Copy.copyGrid(self.hyperbola.grid_y)
        z = Copy.copyGrid(self.hyperbola.grid_z)
        
        nx = Copy.copyGrid(self.hyperbola.grid_nx)
        ny = Copy.copyGrid(self.hyperbola.grid_ny)
        nz = Copy.copyGrid(self.hyperbola.grid_nz)
        
        self.hyperbola.translateGrid(trans1, units='m')
        self.hyperbola.set_cRot(cRot1, units='m')
        self.hyperbola.rotateGrid(rot1)
        self.hyperbola.translateGrid(trans2, units='m')
        self.hyperbola.set_cRot(cRot2, units='m')
        self.hyperbola.rotateGrid(rot2)
        self.hyperbola.translateGrid(trans3, units='m')
        self.hyperbola.set_cRot(cRot3, units='m')
        self.hyperbola.rotateGrid(rot3)

        self.hyperbola.homeReflector()

        for xp, xh, yp, yh, zp, zh, nxp, nxh, nyp, nyh, nzp, nzh in zip(x.ravel(), self.hyperbola.grid_x.ravel(),
                                                                        y.ravel(), self.hyperbola.grid_y.ravel(),
                                                                        z.ravel(), self.hyperbola.grid_z.ravel(),
                                                                        nx.ravel(), self.hyperbola.grid_nx.ravel(),
                                                                        ny.ravel(), self.hyperbola.grid_ny.ravel(),
                                                                        nz.ravel(), self.hyperbola.grid_nz.ravel()):
            self.assertAlmostEqual(xp, xh)
            self.assertAlmostEqual(yp, yh)
            self.assertAlmostEqual(zp, zh)
            
            self.assertAlmostEqual(nxp, nxh)
            self.assertAlmostEqual(nyp, nyh)
            self.assertAlmostEqual(nzp, nzh)

if __name__ == "__main__":
    unittest.main()
        
