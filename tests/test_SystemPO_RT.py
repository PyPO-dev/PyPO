import unittest
import numpy as np
import ctypes

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

    def test_runPO_JM(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runPO, self.s.runPO, self.s.runGUIPO]):
            if dev == "GPU" and not self.GPU_flag:
                return
            
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictJM(source["name"], hyperbola["name"],"test_JM", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictJM(source["name"], parabola["name"],"test_JM", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        if "z" in ellipse["name"]:
                            runPODict = self._get_runPODictJM(source["name"], ellipse["name"],"test_JM", device=dev, direction="bwd")
                        else:
                            runPODict = self._get_runPODictJM(source["name"], ellipse["name"],"test_JM", device=dev)

                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
   
                

    def test_runPO_EH(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runPO, self.s.runPO, self.s.runGUIPO]):
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictEH(source["name"], hyperbola["name"],"test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictEH(source["name"], parabola["name"],"test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        if "z" in ellipse["name"]:
                            runPODict = self._get_runPODictEH(source["name"], ellipse["name"],"test_EH", device=dev, direction="bwd")
                        else:
                            runPODict = self._get_runPODictEH(source["name"], ellipse["name"],"test_EH", device=dev)

                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

    def test_runPO_JMEH(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runPO, self.s.runPO, self.s.runGUIPO]):
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictJMEH(source["name"], hyperbola["name"],"test_JM", "test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictJMEH(source["name"], parabola["name"],"test_JM", "test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        if "z" in ellipse["name"]:
                            runPODict = self._get_runPODictJMEH(source["name"], ellipse["name"],"test_JM", "test_EH", device=dev, direction="bwd")
                        else:
                            runPODict = self._get_runPODictJMEH(source["name"], ellipse["name"],"test_JM", "test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    def test_runPO_EHP_Hybrid(self):
        for interp, dev in zip([False, True], ["CPU", "GPU"]): 
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictEHP(source["name"], hyperbola["name"],"test_EH", "test_P", device=dev)
                        
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)

                        runHybridDict = self._get_runPODictHybrid("test_P", "test_EH", plane["name"], "test_fr_out", "test_field_out", interp=interp)
                         
                        self.s.runHybridPropagation(runHybridDict)
                        self.assertEqual(type(self.s.fields["test_field_out"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_fr_out"]), pypotypes.frame)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictEHP(source["name"], parabola["name"],"test_EH", "test_P", device=dev)
                        
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
                        
                        runHybridDict = self._get_runPODictHybrid("test_P", "test_EH", plane["name"], "test_fr_out", "test_field_out", interp=interp)
                       
                        self.s.runHybridPropagation(runHybridDict)
                        self.assertEqual(type(self.s.fields["test_field_out"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_fr_out"]), pypotypes.frame)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictEHP(source["name"], ellipse["name"],"test_EH", "test_P", device=dev)
                        
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
    
                        runHybridDict = self._get_runPODictHybrid("test_P", "test_EH", plane["name"], "test_fr_out", "test_field_out", interp=interp)
                       
                        self.s.runHybridPropagation(runHybridDict)
                        self.assertEqual(type(self.s.fields["test_field_out"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_fr_out"]), pypotypes.frame)
            

    def test_runPO_FF(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runPO, self.s.runPO, self.s.runGUIPO]):
            if dev == "GPU" and not self.GPU_flag:
                return
            for i, plane in enumerate(TestTemplates.getPlaneList()):
                if i == 2:
                    break

                for source in TestTemplates.getPOSourceList():
                    runPODict = self._get_runPODictFF(source["name"], TestTemplates.getPlaneList()[-1]["name"],"test_EH", device=dev)
                    
                    check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                    
                    func(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    def test_runPO_Scalar(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runPO, self.s.runPO, self.s.runGUIPO]):
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictScalar(source["name"], hyperbola["name"],"test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictScalar(source["name"], parabola["name"],"test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictScalar(source["name"], ellipse["name"],"test_EH", device=dev)
                        
                        check_runPODict(runPODict, self.s.system.keys(), self.s.fields.keys(), self.s.currents.keys(),
                                    self.s.scalarfields.keys(), self.s.frames.keys(), self.s.clog)
                        
                        func(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)

    def test_runRT(self):
        for dev, func in zip(["CPU", "GPU", "CPU"], [self.s.runRayTracer, self.s.runRayTracer, self.s.runGUIRayTracer]):
            if dev == "GPU" and not self.GPU_flag:
                return
            self.s.translateGrids(TestTemplates.TubeRTframe["name"], np.array([0, 0, -1]), obj="frame")
            self.s.translateGrids(TestTemplates.GaussRTframe["name"], np.array([0, 0, -1]), obj="frame")
            
            for hyperbola in TestTemplates.getHyperboloidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], hyperbola["name"],"test_fr", device=dev)
        
                check_runRTDict(runRTDict, self.s.system, self.s.frames, self.s.clog)

                func(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], hyperbola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)

            for parabola in TestTemplates.getParaboloidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], parabola["name"],"test_fr", device=dev)
        
                check_runRTDict(runRTDict, self.s.system, self.s.frames, self.s.clog)

                func(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], parabola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
            
            for ellipse in TestTemplates.getEllipsoidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], ellipse["name"],"test_fr", device=dev)
        
                check_runRTDict(runRTDict, self.s.system, self.s.frames, self.s.clog)

                func(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], ellipse["name"],"test_fr", device=dev)
        
                check_runRTDict(runRTDict, self.s.system, self.s.frames, self.s.clog)

                func(runRTDict)
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

    def test_autoConverge(self): 
        for source in TestTemplates.getPOSourceList():
            gridsize = self.s.autoConverge(source["name"], TestTemplates.hyperboloid_man_xy["name"])

            self.assertEqual(type(gridsize), np.ndarray)
            self.assertEqual(gridsize[0], self.s.system[TestTemplates.hyperboloid_man_xy["name"]]["gridsize"][0])
            self.assertEqual(gridsize[1], self.s.system[TestTemplates.hyperboloid_man_xy["name"]]["gridsize"][1])

            gridsize = self.s.autoConverge(source["name"], TestTemplates.hyperboloid_man_uv["name"])

            self.assertEqual(type(gridsize), np.ndarray)
            self.assertEqual(gridsize[0], self.s.system[TestTemplates.hyperboloid_man_uv["name"]]["gridsize"][0])
            self.assertEqual(gridsize[1], self.s.system[TestTemplates.hyperboloid_man_uv["name"]]["gridsize"][1])


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
    
    def _get_runPODictHybrid(self, fr_in, field_in, target, fr_out, field_out, interp=False):
        runPODict = {
                "fr_in"     : fr_in,
                "t_name"    : target,
                "field_in"  : field_in,
                "fr_out"    : fr_out,
                "field_out" : field_out,
                "interp"    : interp,
                "comp"      : "Ex"
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
    unittest.main()

