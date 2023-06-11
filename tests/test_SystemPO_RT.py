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

from PyPO.System import System

##
# @file
# Tests for checking if CPU and GPU operations in PyPO are correct

class Test_SystemPO_RT(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        
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
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictJM(source["name"], hyperbola["name"],"test_JM", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictJM(source["name"], parabola["name"],"test_JM", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictJM(source["name"], ellipse["name"],"test_JM", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
    
    def test_runPO_EH(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictEH(source["name"], hyperbola["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictEH(source["name"], parabola["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictEH(source["name"], ellipse["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

    def test_runPO_JMEH(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictJMEH(source["name"], hyperbola["name"],"test_JM", "test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictJMEH(source["name"], parabola["name"],"test_JM", "test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictJMEH(source["name"], ellipse["name"],"test_JM", "test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    def test_runPO_EHP(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictEHP(source["name"], hyperbola["name"],"test_EH", "test_P", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictEHP(source["name"], parabola["name"],"test_EH", "test_P", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictEHP(source["name"], ellipse["name"],"test_EH", "test_P", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                        self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
    
    def test_runPO_FF(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            for i, plane in enumerate(TestTemplates.getPlaneList()):
                if i == 2:
                    break

                for source in TestTemplates.getPOSourceList():
                    runPODict = self._get_runPODictFF(source["name"], TestTemplates.getPlaneList()[-1]["name"],"test_EH", device=dev)
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    def test_runPO_Scalar(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            for plane in TestTemplates.getPlaneList():
                for source in TestTemplates.getPOSourceList():
                    for hyperbola in TestTemplates.getHyperboloidList():
                        runPODict = self._get_runPODictScalar(source["name"], hyperbola["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)

                    for parabola in TestTemplates.getParaboloidList():
                        runPODict = self._get_runPODictScalar(source["name"], parabola["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)
                    
                    for ellipse in TestTemplates.getEllipsoidList():
                        runPODict = self._get_runPODictScalar(source["name"], ellipse["name"],"test_EH", device=dev)
                        self.s.runPO(runPODict)
                        self.assertEqual(type(self.s.scalarfields["test_EH"]), pypotypes.scalarfield)

    def test_runRT(self):
        for dev in ["CPU", "GPU"]:
            if dev == "GPU" and not self.GPU_flag:
                return
            self.s.translateGrids(TestTemplates.TubeRTframe["name"], np.array([0, 0, -1]), obj="frame")
            self.s.translateGrids(TestTemplates.GaussRTframe["name"], np.array([0, 0, -1]), obj="frame")
            
            for hyperbola in TestTemplates.getHyperboloidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], hyperbola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], hyperbola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)

            for parabola in TestTemplates.getParaboloidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], parabola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], parabola["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
            
            for ellipse in TestTemplates.getEllipsoidList():
                runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], ellipse["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
                
                runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], ellipse["name"],"test_fr", device=dev)
                self.s.runRayTracer(runRTDict)
                self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
        
    def _get_runPODictJM(self, source_current, target, name_JM, device="CPU"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : device,
                "name_JM"   : name_JM,
                "mode"      : "JM"
                }

        return runPODict
    
    def _get_runPODictEH(self, source_current, target, name_EH, device="CPU"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : device,
                "name_EH"   : name_EH,
                "mode"      : "EH"
                }

        return runPODict
    
    def _get_runPODictJMEH(self, source_current, target, name_JM, name_EH, device="CPU"):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
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

