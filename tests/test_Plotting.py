"""!
@file
This file contains tests for the plotting functionalities. It tests whether the plotting functions in the System behave as expected.
"""

import unittest
import warnings
import sys

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError: 
    import TestTemplates

from numpy import ndarray 
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes, close

from mpl_toolkits.mplot3d.axes3d import Axes3D
from PyPO.System import System
from PyPO.Enums import Projections, FieldComponents, CurrentComponents, Modes

class Test_Plotting(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False)

    @params(Modes.dB, Modes.LIN)
    def test_plotBeamCut(self, mode):
        fig, ax = self.s.plotBeamCut(TestTemplates.GPOfield['name'], FieldComponents.Ex, center=False, align=False, ret=True, mode=mode)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes)
        
        close('all')

    @params(Projections.xy, Projections.yz, Projections.zx,
            Projections.yx, Projections.zy, Projections.xz)
    def test_plotBeam2D(self, project):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            aper_l = [TestTemplates.aperDictEll, TestTemplates.aperDictRect]

            for aper in aper_l:
                out_ar = []
                out_ax = []
                aperDict_plot = self.s.copyObj(aper)
                aperDict_plot["plot"] = True

                out_ax.append(self.s.plotBeam2D(TestTemplates.GPOfield['name'], FieldComponents.Ex, 
                    ret=True, amp_only=True, project=project, aperDict=aperDict_plot))
                
                out_ax.append(self.s.plotBeam2D(TestTemplates.GPOfield['name'], FieldComponents.Ex, 
                    ret=True, amp_only=True, mode=Modes.LIN, project=project))

                out_ar.append(self.s.plotBeam2D(TestTemplates.GPOfield['name'], FieldComponents.Ex, 
                    ret=True, project=project, aperDict=aper, contour=TestTemplates.GPOfield['name'], 
                    contour_comp=FieldComponents.Ex, levels=[0.5, 1]))
                
                out_ar.append(self.s.plotBeam2D(TestTemplates.GPOfield['name'], FieldComponents.Ex, 
                    ret=True, mode=Modes.LIN, project=project, contour=TestTemplates.GPOfield['name'], 
                    contour_comp=FieldComponents.Ex, levels=[0.5, 1]))
                
                out_ax.append(self.s.plotBeam2D(TestTemplates.PS_Ufield_FF["name"], FieldComponents.Ex, ret=True, project=Projections.xy))
                out_ax.append(self.s.plotBeam2D(TestTemplates.PS_Ufield_FF["name"], FieldComponents.Ex, ret=True, amp_only=True, project=Projections.xy))

            for entry_ax, entry_ar in zip(out_ax, out_ar):
                self.assertEqual(type(entry_ax[0]), Figure)
                self.assertEqual(type(entry_ar[0]), Figure)
               
                self.assertEqual(type(entry_ax[1]), Axes)
                self.assertEqual(type(entry_ar[1]), ndarray)
            
            close('all')

    @params(*TestTemplates.getAllSurfList())
    def test_plot3D(self, element):
        sys.tracebacklimit = 0
        try:
            fig, ax = self.s.plot3D(element['name'], ret=True, foc1=True, foc2=True, norm=True)

        except KeyError:
            fig, ax = self.s.plot3D(element['name'], ret=True, norm=True)
            
        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes3D)

        close('all')

            
    def test_plotSystem(self):
        fig, ax = self.s.plotSystem(ret=True)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes3D)

        emptySys = System(verbose=False)
        figE, axE = emptySys.plotSystem(ret=True)

        self.assertEqual(type(figE), Figure)
        self.assertEqual(type(axE), Axes3D)
        
        figRT, axRT = self.s.plotSystem(ret=True, RTframes=[TestTemplates.TubeRTframe["name"]])

        self.assertEqual(type(figRT), Figure)
        self.assertEqual(type(axRT), Axes3D)

        close('all')
        
        
    def test_plotGroup(self):
        self.s.groupElements('testGroup', 
                          TestTemplates.paraboloid_man_xy['name'],
                          TestTemplates.hyperboloid_man_uv['name'],
                          TestTemplates.ellipsoid_z_foc_xy['name'],
                          TestTemplates.ellipsoid_x_man_uv['name'],
                          TestTemplates.plane_xy['name'],
                          TestTemplates.plane_AoE['name'],
                          )
        
        fig, ax = self.s.plotGroup('testGroup', ret=True)

        self.assertEqual(type(fig), Figure)
        self.assertEqual(type(ax), Axes3D)

        close('all')
        
        
    @params(Projections.xy, Projections.yz, Projections.zx,
            Projections.yx, Projections.zy, Projections.xz)
    def test_plotRTframe(self, project):
        for frameName in [TestTemplates.TubeRTframe['name'], TestTemplates.GaussRTframe['name']]:
            fig = self.s.plotRTframe(frameName, ret=True, project=project)

            self.assertEqual(type(fig), Figure)

            close('all')        

if __name__ == '__main__':
    import nose2
    nose2.main()
