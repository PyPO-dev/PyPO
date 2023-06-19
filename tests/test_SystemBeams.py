import unittest
import ctypes

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import PyPO.BindBeam as beamlibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if beams in PyPO are correct

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

if __name__ == "__main__":
    import nose2
    nose2.main()
