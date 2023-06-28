import unittest
import numpy as np
import ctypes

from nose2.tools import params

try:
    from . import TestTemplates
except:
    import TestTemplates

import PyPO.BindCPU as cpulibs
import PyPO.BindGPU as gpulibs
import PyPO.PyPOTypes as pypotypes
import PyPO.Templates as pytemp

from PyPO.System import System
from PyPO.Checks import RunPOError, RunRTError, HybridPropError, check_runPODict, check_hybridDict, check_runRTDict

##
# @file
# Tests for checking if CPU and GPU operations in PyPO are correct

class Test_SystemPO_RT(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setLoggingVerbosity(False)
        self.GPU_flag = True
        try:
            lib = gpulibs.loadGPUlib()
        except OSError:
            self.GPU_flag = False

    def test_loadCPUlib(self):
        lib = cpulibs.loadCPUlib()
        self.assertEqual(type(lib), ctypes.CDLL)
        
    def test_loadGPUlib(self):
        if self.GPU_flag:
            lib = gpulibs.loadGPUlib()
            self.assertEqual(type(lib), ctypes.CDLL)
        else:
            print("No GPU libraries found... Not testing GPU functionalities.")
    
    @params(("CPU", System.runPO), ("GPU", System.runPO), ("CPU", System.runGUIPO))
    def test_runPO_JM(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
      
        for direction in ["fwd", "bwd"]:
            runPODict = self._get_runPODictJM(TestTemplates.GPOfield["name"], TestTemplates.paraboloid_man_xy["name"],"test_JM", device=dev, direction=direction)
            
            check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                        self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                
            func(self.s, runPODict)
            self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents) 
                
    @params(("CPU", System.runPO), ("GPU", System.runPO), ("CPU", System.runGUIPO))
    def test_runPO_EH(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
        
        for direction in ["fwd", "bwd"]:
            runPODict = self._get_runPODictEH(TestTemplates.GPOfield["name"], TestTemplates.paraboloid_man_xy["name"],"test_EH", device=dev, direction=direction)
            
            check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                        self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                
            func(self.s, runPODict)
            self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

    @params(("CPU", System.runPO), ("GPU", System.runPO), ("CPU", System.runGUIPO))
    def test_runPO_JMEH(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
        
        for direction in ["fwd", "bwd"]:
            runPODict = self._get_runPODictJMEH(TestTemplates.GPOfield["name"], TestTemplates.paraboloid_man_xy["name"], "test_JM", "test_EH", device=dev, direction=direction)
            
            check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                        self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                
            func(self.s, runPODict)
            self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
            self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    @params(("CPU", True, System.runPO, System.runHybridPropagation, None, np.ones(3)), 
            ("GPU", True, System.runPO, System.runHybridPropagation, "Ex", None), 
            ("CPU", False, System.runGUIPO, System.hybridGUIPropagation, "Ex", None))
    def test_runPO_EHP_Hybrid(self, dev, interp, funcPO, funcHybrid, comp, start):
        if dev == "GPU" and not self.GPU_flag:
            return

        runPODict = self._get_runPODictEHP(TestTemplates.GPOfield["name"], TestTemplates.paraboloid_man_xy["name"],"test_EH", "test_P", device=dev)
            
        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
        
        funcPO(self.s, runPODict)
        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
        
        runHybridDict = self._get_runPODictHybrid("test_P", "test_EH", TestTemplates.paraboloid_man_xy["name"], "test_fr_out", "test_field_out", interp=interp, comp=comp, start=start)
        
        check_hybridDict(runHybridDict, self.s.system.keys(), self.s.frames.keys(), self.s.fields.keys(), self.s.clog)
        
        funcHybrid(self.s, runHybridDict)
        self.assertEqual(type(self.s.fields["test_field_out"]), pypotypes.fields)
        self.assertEqual(type(self.s.frames["test_fr_out"]), pypotypes.frame)
            
    @params(("CPU", System.runPO), ("GPU", System.runPO), ("CPU", System.runGUIPO))
    def test_runPO_FF(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
        
        runPODict = self._get_runPODictFF(TestTemplates.GPOfield["name"], TestTemplates.getPlaneList()[-1]["name"], "test_EH", device=dev)
        
        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
        
        func(self.s, runPODict)
        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    @params(("CPU", System.runPO), ("GPU", System.runPO), ("CPU", System.runGUIPO))
    def test_runPO_Scalar(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
                    
        runPODict = self._get_runPODictScalar(TestTemplates.GPOfield["name"], TestTemplates.paraboloid_man_xy["name"],"test_EH", device=dev)
        
        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
        
        func(self.s, runPODict)
        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)

    @params(("CPU", System.runRayTracer), ("GPU", System.runRayTracer), ("CPU", System.runGUIRayTracer))
    def test_runRT(self, dev, func):
        if dev == "GPU" and not self.GPU_flag:
            return
        self.s.translateGrids(TestTemplates.TubeRTframe["name"], np.array([0, 0, -1]), obj="frame")
        self.s.translateGrids(TestTemplates.GaussRTframe["name"], np.array([0, 0, -1]), obj="frame")
            
        runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], TestTemplates.paraboloid_man_xy["name"],"test_fr", device=dev)

        check_runRTDict(runRTDict, self.s.system, self.s.frames, self.s.clog)

        func(self.s, runRTDict)
        self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)

    def test_invalidRunPODict(self):
        testDict = {}
        
        with self.assertRaises(RunPOError):
            self.s.runPO(testDict)

        badDict = pytemp.runPODict
        
        with self.assertRaises(RunPOError):
            self.s.runPO(badDict)
    
    def test_invalidRunRTDict(self):
        testDict = {}

        with self.assertRaises(RunRTError):
            self.s.runRayTracer(testDict)
        
        badDict = pytemp.runRTDict
        
        with self.assertRaises(RunRTError):
            self.s.runRayTracer(badDict)

    def test_invalidRunHybridDict(self):
        testDict = {}

        with self.assertRaises(HybridPropError):
            self.s.runHybridPropagation(testDict)
        
        badDict = pytemp.hybridDict
        
        with self.assertRaises(HybridPropError):
            self.s.runHybridPropagation(badDict)

    @params(*TestTemplates.getPOSourceList())
    def test_autoConverge(self, source): 
        gridsize = self.s.autoConverge(source["name"], TestTemplates.hyperboloid_man_xy["name"])

        self.assertEqual(type(gridsize), np.ndarray)
        self.assertEqual(gridsize[0], self.s.system[TestTemplates.hyperboloid_man_xy["name"]]["gridsize"][0])
        self.assertEqual(gridsize[1], self.s.system[TestTemplates.hyperboloid_man_xy["name"]]["gridsize"][1])

        gridsize = self.s.autoConverge(source["name"], TestTemplates.hyperboloid_man_uv["name"])

        self.assertEqual(type(gridsize), np.ndarray)
        self.assertEqual(gridsize[0], self.s.system[TestTemplates.hyperboloid_man_uv["name"]]["gridsize"][0])
        self.assertEqual(gridsize[1], self.s.system[TestTemplates.hyperboloid_man_uv["name"]]["gridsize"][1])

    def test_mergeBeams(self):
        self.s.mergeBeams(TestTemplates.GPOfield["name"], TestTemplates.PS_Ufield["name"], merged_name="test")
        self.s.mergeBeams(TestTemplates.GPOfield["name"], TestTemplates.PS_Ufield["name"], obj="currents", merged_name="test", )
        self.assertTrue("test" in self.s.fields)
        self.assertTrue("test" in self.s.currents)
    
    @params(*TestTemplates.getPOSourceList())
    def test_interpBeam(self, source):
        name = source["name"]
        gridsize_new = np.array([101, 101])
        self.s.interpBeam(name, gridsize_new, obj_t="fields")
        self.s.interpBeam(name, gridsize_new, obj_t="currents")
    
        self.assertTrue(f"{name}_interp" in self.s.fields)
        self.assertTrue(f"{name}_interp" in self.s.currents)

    def _get_runPODictJM(self, source_current, target, name_JM, device="CPU", direction="fwd"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : direction,
                "device"    : device,
                "name_JM"   : name_JM,
                "mode"      : "JM"
                }

        return runPODict
    
    def _get_runPODictEH(self, source_current, target, name_EH, device="CPU", direction="fwd"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : direction,
                "device"    : device,
                "name_EH"   : name_EH,
                "mode"      : "EH"
                }

        return runPODict
    
    def _get_runPODictJMEH(self, source_current, target, name_JM, name_EH, device="CPU", direction="fwd"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : direction,
                "device"    : device,
                "name_JM"   : name_JM,
                "name_EH"   : name_EH,
                "mode"      : "JMEH"
                }

        return runPODict
    
    def _get_runPODictFF(self, source_current, target, name_EH, device="CPU"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : device,
                "name_EH"   : name_EH,
                "mode"      : "FF"
                }
    
        return runPODict

    def _get_runPODictEHP(self, source_current, target, name_EH, name_P, device="CPU"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : device,
                "name_EH"   : name_EH,
                "name_P"    : name_P,
                "mode"      : "EHP"
                }

        return runPODict
    
    def _get_runPODictHybrid(self, fr_in, field_in, target, fr_out, field_out, interp=False, comp=None, start=None):
        runPODict = {
                "fr_in"     : fr_in,
                "t_name"    : target,
                "field_in"  : field_in,
                "fr_out"    : fr_out,
                "field_out" : field_out,
                "interp"    : interp,
                "comp"      : comp,
                "start"     : start
                }

        return runPODict
    
    def _get_runPODictScalar(self, source_field, target, name_scalarfield, device="CPU"):
        runPODict = {
                "t_name"        : target,
                "s_scalarfield" : source_field,
                "epsilon"       : 10,
                "exp"           : "fwd",
                "device"        : device,
                "name_field"    : name_scalarfield,
                "mode"          : "scalar"
                }

        return runPODict

    def _get_runRTDict(self, fr_in, target, fr_out, device="CPU"):
        runRTDict = {
                "fr_in"     : fr_in,
                "t_name"    : target,
                "device"    : device,
                "fr_out"    : fr_out,
                "t0"        : 1
                }

        return runRTDict

if __name__ == "__main__":
    import nose2
    nose2.main()

