import unittest
import numpy as np
from PyPO.Checks import InputReflError, InputRTError, InputPOError
import PyPO.Templates as pypotemp

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

        self.invalidDict = {'name' : 'someElement'}
        
    def test_dictsExist(self):
        for i in ["system", "groups", "frames", "currents", "fields", "scalarfields"]:
            self.assertIsNotNone(self.s.__getattribute__(i))

    def test_addReflector(self):
        ltot = len(TestTemplates.getParaboloidList()) + len(TestTemplates.getHyperboloidList()) + len(TestTemplates.getPlaneList()) + len(TestTemplates.getEllipsoidList())

        self.assertEqual(len(self.s.system), ltot)

        with self.assertRaises(InputReflError):
            self.s.addPlane(self.invalidDict) 
            self.s.addParabola(self.invalidDict) 
            self.s.addHyperbola(self.invalidDict) 
            self.s.addEllipse(self.invalidDict) 

            self.s.addPlane(pypotemp.reflDict) 
            self.s.addParabola(pypotemp.reflDict) 
            self.s.addHyperbola(pypotemp.reflDict) 
            self.s.addEllipse(pypotemp.reflDict) 


    def test_addGroup(self):
        ltot = 0
        with self.assertRaises(TypeError):
            self.s.groupElements()
        self.assertEqual(len(self.s.groups), ltot)
        
        self.s.groupElements(TestTemplates.paraboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"])
        self.assertEqual(len(self.s.groups), ltot + 1)

    def test_addRTFrame(self):
        ltot = len(TestTemplates.getFrameList())

        self.assertEqual(len(self.s.frames), ltot)

        with self.assertRaises(InputRTError):
            self.s.createTubeFrame(self.invalidDict) 
            self.s.createGRTFrame(self.invalidDict) 
            
            self.s.createTubeFrame(pypotemp.TubeRTDict) 
            self.s.createGRTFrame(pypotemp.GRTDict) 

    def test_addPOFields(self):
        ltot = len(TestTemplates.getPOSourceList())

        self.assertEqual(len(self.s.fields), ltot)
        self.assertEqual(len(self.s.currents), ltot)

        with self.assertRaises(InputPOError):
            self.s.createGaussian(self.invalidDict, TestTemplates.plane_xy["name"]) 
            self.s.createPointSource(self.invalidDict, TestTemplates.plane_xy["name"]) 
            self.s.createUniformSource(self.invalidDict, TestTemplates.plane_xy["name"]) 

            self.s.createGaussian(pypotemp.GPODict, TestTemplates.plane_xy["name"]) 
            self.s.createPointSource(pypotemp.UPSDict, TestTemplates.plane_xy["name"]) 
            self.s.createUniformSource(pypotemp.UPSDict, TestTemplates.plane_xy["name"]) 
    
    def test_addPOScalarFields(self):
        ltot = len(TestTemplates.getPOSourceList())

        self.assertEqual(len(self.s.scalarfields), ltot)

        with self.assertRaises(InputPOError):
            self.s.createScalarGaussian(self.invalidDict, TestTemplates.plane_xy["name"]) 
            self.s.createPointSourceScalar(self.invalidDict, TestTemplates.plane_xy["name"]) 
            self.s.createUniformSourceScalar(self.invalidDict, TestTemplates.plane_xy["name"]) 
    
            self.s.createScalarGaussian(pypotemp.GPODict, TestTemplates.plane_xy["name"]) 
            self.s.createPointSourceScalar(pypotemp.UPSDict, TestTemplates.plane_xy["name"]) 
            self.s.createUniformSourceScalar(pypotemp.UPSDict, TestTemplates.plane_xy["name"]) 

if __name__ == "__main__":
    unittest.main()
        
