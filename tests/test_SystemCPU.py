import unittest
import numpy as np
import ctypes

try:
    from . import TestTemplates
except:
    import TestTemplates

import PyPO.BindCPU as cpulibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

##
# @file
# Tests for checking if CPU operations in PyPO are correct

class Test_SystemCPU(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

    def test_loadCPUlib(self):
        lib = cpulibs.loadCPUlib()
        self.assertEqual(type(lib), ctypes.CDLL)

    def test_runPO_JM(self):
        for plane in TestTemplates.getPlaneList():
            for i, source in enumerate(TestTemplates.getPOSourceList()):
                if i == 0:
                    self.s.createGaussian(source, plane["name"])
                elif i == 1:
                    self.s.createPointSource(source, plane["name"])
                for hyperbola in TestTemplates.getHyperboloidList():
                    runPODict = self._get_runPODictJM(source["name"], hyperbola["name"],"test_JM")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)

                for parabola in TestTemplates.getParaboloidList():
                    runPODict = self._get_runPODictJM(source["name"], parabola["name"],"test_JM")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                
                for ellipse in TestTemplates.getEllipsoidList():
                    runPODict = self._get_runPODictJM(source["name"], ellipse["name"],"test_JM")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
    
    def test_runPO_EH(self):
        for plane in TestTemplates.getPlaneList():
            for i, source in enumerate(TestTemplates.getPOSourceList()):
                if i == 0:
                    self.s.createGaussian(source, plane["name"])
                elif i == 1:
                    self.s.createPointSource(source, plane["name"])
                for hyperbola in TestTemplates.getHyperboloidList():
                    runPODict = self._get_runPODictEH(source["name"], hyperbola["name"],"test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                for parabola in TestTemplates.getParaboloidList():
                    runPODict = self._get_runPODictEH(source["name"], parabola["name"],"test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                
                for ellipse in TestTemplates.getEllipsoidList():
                    runPODict = self._get_runPODictEH(source["name"], ellipse["name"],"test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

    def test_runPO_JMEH(self):
        for plane in TestTemplates.getPlaneList():
            for i, source in enumerate(TestTemplates.getPOSourceList()):
                if i == 0:
                    self.s.createGaussian(source, plane["name"])
                
                elif i == 1:
                    self.s.createPointSource(source, plane["name"])

                for hyperbola in TestTemplates.getHyperboloidList():
                    runPODict = self._get_runPODictJMEH(source["name"], hyperbola["name"],"test_JM", "test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

                for parabola in TestTemplates.getParaboloidList():
                    runPODict = self._get_runPODictJMEH(source["name"], parabola["name"],"test_JM", "test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                
                for ellipse in TestTemplates.getEllipsoidList():
                    runPODict = self._get_runPODictJMEH(source["name"], ellipse["name"],"test_JM", "test_EH")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.currents["test_JM"]), pypotypes.currents)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
    
    def test_runPO_EHP(self):
        for plane in TestTemplates.getPlaneList():
            for i, source in enumerate(TestTemplates.getPOSourceList()):
                if i == 0:
                    self.s.createGaussian(source, plane["name"])
                
                elif i == 1:
                    self.s.createPointSource(source, plane["name"])
                    
                for hyperbola in TestTemplates.getHyperboloidList():
                    runPODict = self._get_runPODictEHP(source["name"], hyperbola["name"],"test_EH", "test_P")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)

                for parabola in TestTemplates.getParaboloidList():
                    runPODict = self._get_runPODictEHP(source["name"], parabola["name"],"test_EH", "test_P")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
                
                for ellipse in TestTemplates.getEllipsoidList():
                    runPODict = self._get_runPODictEHP(source["name"], ellipse["name"],"test_EH", "test_P")
                    self.s.runPO(runPODict)
                    self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)
                    self.assertEqual(type(self.s.frames["test_P"]), pypotypes.frame)
    
    def test_runPO_FF(self):
        for i, plane in enumerate(TestTemplates.getPlaneList()):
            if i == 2:
                break

            for i, source in enumerate(TestTemplates.getPOSourceList()):
                if i == 0:
                    self.s.createGaussian(source, plane["name"])
                
                elif i == 1:
                    self.s.createPointSource(source, plane["name"])
                    
                runPODict = self._get_runPODictFF(source["name"], TestTemplates.getPlaneList()[-1]["name"],"test_EH")
                self.s.runPO(runPODict)
                self.assertEqual(type(self.s.fields["test_EH"]), pypotypes.fields)

    def test_runRT(self):
        self.s.translateGrids(TestTemplates.TubeRTframe["name"], np.array([0, 0, -1]), obj="frame")
        self.s.translateGrids(TestTemplates.GaussRTframe["name"], np.array([0, 0, -1]), obj="frame")
        
        for hyperbola in TestTemplates.getHyperboloidList():
            runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], hyperbola["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
            
            runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], hyperbola["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)

        for parabola in TestTemplates.getParaboloidList():
            runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], parabola["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
            
            runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], parabola["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
        
        for ellipse in TestTemplates.getEllipsoidList():
            runRTDict = self._get_runRTDict(TestTemplates.TubeRTframe["name"], ellipse["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
            
            runRTDict = self._get_runRTDict(TestTemplates.GaussRTframe["name"], ellipse["name"],"test_fr")
            self.s.runRayTracer(runRTDict)
            self.assertEqual(type(self.s.frames["test_fr"]), pypotypes.frame)
        
    def _get_runPODictJM(self, source_current, target, name_JM):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : "CPU",
                "name_JM"   : name_JM,
                "mode"      : "JM"
                }

        return runPODict
    
    def _get_runPODictEH(self, source_current, target, name_EH):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : "CPU",
                "name_EH"   : name_EH,
                "mode"      : "EH"
                }

        return runPODict
    
    def _get_runPODictJMEH(self, source_current, target, name_JM, name_EH):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : "CPU",
                "name_JM"   : name_JM,
                "name_EH"   : name_EH,
                "mode"      : "JMEH"
                }

        return runPODict
    
    def _get_runPODictFF(self, source_current, target, name_EH):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : "CPU",
                "name_EH"   : name_EH,
                "mode"      : "FF"
                }
    
        return runPODict

    def _get_runPODictEHP(self, source_current, target, name_EH, name_P):
        runPODict = {
                "t_name"    : target,
                "s_current" : source_current,
                "epsilon"   : 10,
                "exp"       : "fwd",
                "device"    : "CPU",
                "name_EH"   : name_EH,
                "name_P"    : name_P,
                "mode"      : "EHP"
                }

        return runPODict

    def _get_runRTDict(self, fr_in, target, fr_out):
        runRTDict = {
                "fr_in"     : fr_in,
                "t_name"    : target,
                "device"    : "CPU",
                "fr_out"    : fr_out,
                "t0"        : 1
                }

        return runRTDict

if __name__ == "__main__":
    unittest.main()

