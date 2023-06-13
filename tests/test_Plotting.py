import unittest
import warnings
try:
    from . import TestTemplates
except ImportError: 
    import TestTemplates

from numpy import ndarray 
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes, close
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PyPO.System import System

##
# @file
# This file contains tests for the plotting functionalities. It tests whether the plotting functions in the System behave as expected.


stm = TestTemplates.getSystemWithReflectors()

class Test_SystemOps(unittest.TestCase):

    def test_plotBeamCut(self):
        fig, ax = stm.plotBeamCut(TestTemplates.GPOfield['name'], 'Ex', center=False, align=False, ret=True)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes)

        close('all')

    def test_plotBeam2D(self):
        out_ar = []
        out_ax = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_ax.append(stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret=True, amp_only=True))
            out_ax.append(stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret=True, amp_only=True, mode="linear"))
            out_ar.append(stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret=True))
            out_ar.append(stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret=True, mode="linear"))

        for entry_ax, entry_ar in zip(out_ax, out_ar):
            self.assertEqual(type(entry_ax[0]), Figure)
            self.assertEqual(type(entry_ar[0]), Figure)
            
            self.assertEqual(type(entry_ax[1]), Axes)
            self.assertEqual(type(entry_ar[1]), ndarray)

        close('all')

    def test_plotBeam3D(self):
        fig1, ax1 = stm.plot3D(TestTemplates.paraboloid_man_uv['name'], ret=True)
        self.assertEqual(type(fig1), Figure)
        self.assertEqual(type(ax1), Axes3D)

        for refl in (TestTemplates.getPlaneList() + 
                     TestTemplates.getParaboloidList() + 
                     TestTemplates.getHyperboloidList() +
                     TestTemplates.getEllipsoidList()):
            fig, ax = stm.plot3D(refl['name'], ret=True)

            self.assertEqual(type(fig), Figure)
            self.assertEqual(type(ax), Axes3D)

            close('all')

            
    def test_plotSystem(self):
        fig, ax = stm.plotSystem(ret=True)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes3D)

        emptySys = System(verbose=False)
        figE, axE = emptySys.plotSystem(ret=True)

        self.assertEqual(type(figE), Figure)
        self.assertEqual(type(axE), Axes3D)
        

        close('all')
        
        
    def test_plotGroup(self):
        stm.groupElements('testGroup', 
                          TestTemplates.paraboloid_man_xy['name'],
                          TestTemplates.hyperboloid_man_uv['name'],
                          TestTemplates.ellipsoid_z_foc_xy['name'],
                          TestTemplates.ellipsoid_x_man_uv['name'],
                          TestTemplates.plane_xy['name'],
                          TestTemplates.plane_AoE['name'],
                          )
        
        fig, ax = stm.plotGroup('testGroup', ret=True)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes3D)

        close('all')
        
        
    def test_plotRTframe(self):
        for frameName in [TestTemplates.TubeRTframe['name'], TestTemplates.GaussRTframe['name']]:
            fig = stm.plotRTframe(frameName, ret=True)

            self.assertEqual(type(fig), Figure)

            close('all')        

if __name__ == '__main__':
    unittest.main()


