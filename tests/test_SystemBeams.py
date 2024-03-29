"""!
@file
Tests for checking if beams in PyPO are correct
"""

import unittest
import ctypes
import numpy as np
import os

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import PyPO.BindBeam as beamlibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System
from PyPO.Enums import FieldComponents, CurrentComponents

class Test_SystemBeams(unittest.TestCase):
    
    ##
    # Determine whether shared object exists or not.
    def _get_lib_exis(self, module, libname):
        path_cur = pathlib.Path(module.__file__).parent.resolve()
        try:
            lib = os.path.join(path_cur, f"{libname}.dll")
        except:
            try:
                lib = os.path.join(path_cur, f"{libname}.so")
            except:
                lib = os.path.join(path_cur, f"{libname}.dylib")

        flag = os.path.exists(lib)

        return flag

    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

    def test_loadBeamlib(self):
        lib = beamlibs.loadBeamlib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_createRTframe(self):
        self.s.createTubeFrame(TestTemplates.TubeRTframe)
        self.assertEqual(type(self.s.frames["testTubeRTframe"]), pypotypes.frame)

    def test_createGRTframe(self):
        self.s.createGRTFrame(TestTemplates.GaussRTframe)
        self.assertEqual(type(self.s.frames["testGaussRTframe"]), pypotypes.frame)

    @params(*TestTemplates.getReflectorPlaneList())
    def test_createGauss(self, plane):
        self.s.createGaussian(TestTemplates.GPOfield, plane["name"])
        self.assertEqual(type(self.s.fields["testGaussField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testGaussField"]), pypotypes.currents)

    @params(*TestTemplates.getReflectorPlaneList())
    def test_createScalarGauss(self, plane):
        self.s.createScalarGaussian(TestTemplates.GPOfield, plane["name"])
        self.assertEqual(type(self.s.scalarfields["testGaussField"]), pypotypes.scalarfield)

    @params(*TestTemplates.getReflectorPlaneList())
    def test_createPointSource(self, plane):
        self.s.createPointSource(TestTemplates.PS_Ufield, plane["name"])
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)
    
    @params(*TestTemplates.getReflectorPlaneList())
    def test_createScalarPointSource(self, plane):
        self.s.createPointSourceScalar(TestTemplates.PS_Ufield, plane["name"])
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)
    
    @params(*TestTemplates.getReflectorPlaneList())
    def test_createUniformSource(self, plane):
        self.s.createUniformSource(TestTemplates.PS_Ufield, plane["name"])
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)
    
    @params(*TestTemplates.getReflectorPlaneList())
    def test_createScalarUniformSource(self, plane):
        self.s.createUniformSourceScalar(TestTemplates.PS_Ufield, plane["name"])
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)

    def test_readCustomBeam(self):
        root = os.path.dirname(os.path.realpath(__file__))
        rcbeamPath = os.path.join(root, "rtest_cbeam.txt")
        icbeamPath = os.path.join(root, "itest_cbeam.txt")
        np.savetxt(rcbeamPath, np.ones((13, 13)))
        np.savetxt(icbeamPath, np.ones((13, 13)))

        self.s.setCustomBeamPath(root)
        self.s.readCustomBeam("test_cbeam", TestTemplates.plane_xy["name"], FieldComponents.Ex, lam=1)
        self.assertTrue("test_cbeam" in self.s.fields)
        self.assertTrue("test_cbeam" in self.s.currents)

        os.remove(rcbeamPath)
        os.remove(icbeamPath)

if __name__ == "__main__":
    import nose2
    nose2.main()
