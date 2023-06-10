import unittest
import numpy as np
import ctypes

import TestUtils.TestTemplates as TestTemplates

import PyPO.BindBeam as beamlibs
import PyPO.BindCPU as cpulibs
import PyPO.BindRefl as refllibs
import PyPO.BindTransf as transflibs
import PyPO.BindGPU as gpulibs

import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if shared object files are importable

class Test_SystemBindings(unittest.TestCase):
    
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
        self.s = System(verbose=False)

    def test_loadBeamlib(self):
        lib = beamlibs.loadBeamlib()
        self.assertEqual(type(lib), ctypes.CDLL)
    
    def test_loadTransflib(self):
        lib = transflibs.loadTransflib()
        self.assertEqual(type(lib), ctypes.CDLL)
    
    def test_loadRefllib(self):
        lib = refllibs.loadRefllib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_loadCPUlib(self):
        lib = cpulibs.loadCPUlib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_loadGPUlib(self):
        lib = gpulibs.loadGPUlib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_createRTframe(self):
        self.s.createTubeFrame(TestTemplates.TubeRTframe)
        self.assertEqual(type(self.s.frames["testTubeRTframe"]), pypotypes.frame)

    def test_createGRTframe(self):
        self.s.createGRTFrame(TestTemplates.GaussRTframe)
        self.assertEqual(type(self.s.frames["testGaussRTframe"]), pypotypes.frame)

    def test_createGauss(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createGaussian(TestTemplates.GPOfield, "testPlane_xy")
        self.assertEqual(type(self.s.fields["testGaussField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testGaussField"]), pypotypes.currents)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createGaussian(TestTemplates.GPOfield, "testPlane_uv")
        self.assertEqual(type(self.s.fields["testGaussField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testGaussField"]), pypotypes.currents)
    
    def test_createScalarGauss(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createScalarGaussian(TestTemplates.GPOfield, "testPlane_xy")
        self.assertEqual(type(self.s.scalarfields["testGaussField"]), pypotypes.scalarfield)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createScalarGaussian(TestTemplates.GPOfield, "testPlane_uv")
        self.assertEqual(type(self.s.scalarfields["testGaussField"]), pypotypes.scalarfield)

    def test_createPointSource(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createPointSource(TestTemplates.PS_Ufield, "testPlane_xy")
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createPointSource(TestTemplates.PS_Ufield, "testPlane_uv")
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)
    
    def test_makeScalarPointSource(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createPointSourceScalar(TestTemplates.PS_Ufield, "testPlane_xy")
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createPointSourceScalar(TestTemplates.PS_Ufield, "testPlane_uv")
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)
    
    def test_createUniformSource(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createUniformSource(TestTemplates.PS_Ufield, "testPlane_xy")
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createUniformSource(TestTemplates.PS_Ufield, "testPlane_uv")
        self.assertEqual(type(self.s.fields["testPS_UField"]), pypotypes.fields)
        self.assertEqual(type(self.s.currents["testPS_UField"]), pypotypes.currents)
    
    def test_makeScalarUniformSource(self):
        self.s.addPlane(TestTemplates.plane_xy)
        self.s.createUniformSourceScalar(TestTemplates.PS_Ufield, "testPlane_xy")
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)

        self.s.addPlane(TestTemplates.plane_uv)
        self.s.createUniformSourceScalar(TestTemplates.PS_Ufield, "testPlane_uv")
        self.assertEqual(type(self.s.scalarfields["testPS_UField"]), pypotypes.scalarfield)
if __name__ == "__main__":
    unittest.main()
