"""!
@file
Tests for checking if objects are properly added to the system.
"""

import unittest
from PyPO.Checks import InputReflError, InputRTError, InputPOError
import PyPO.Templates as pypotemp

from PyPO.System import System

from nose2.tools import params

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

    @params(System.addPlane, System.addParabola, System.addHyperbola, System.addEllipse)
    def test_addReflector(self, func):
        ltot = len(TestTemplates.getAllReflectorList()) + 1 # 1 extra for far-field plane

        self.assertEqual(len(self.s.system), ltot)

        with self.assertRaises(InputReflError):
            func(self.s, self.invalidDict)

    def test_addGroup(self):
        ltot = 0
        with self.assertRaises(TypeError):
            self.s.groupElements()
        self.assertEqual(len(self.s.groups), ltot)
        
        self.s.groupElements(TestTemplates.paraboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"])
        self.assertEqual(len(self.s.groups), ltot + 1)

    def test_removeGroup(self):
        self.s.groupElements("test", TestTemplates.paraboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"])
        self.assertTrue("test" in self.s.groups)

        self.s.removeGroup("test")
        self.assertFalse("test" in self.s.groups)

    @params(System.createTubeFrame, System.createGRTFrame)
    def test_addRTFrame(self, func):
        ltot = len(TestTemplates.getFrameList())

        self.assertEqual(len(self.s.frames), ltot)

        with self.assertRaises(InputRTError):
            func(self.s, self.invalidDict) 

    @params(System.createGaussian, System.createPointSource, System.createUniformSource)
    def test_addPOFields(self, func):
        ltot = len(TestTemplates.getPOSourceList()) + 1

        self.assertEqual(len(self.s.fields), ltot)
        self.assertEqual(len(self.s.currents), ltot)

        with self.assertRaises(InputPOError):
            func(self.s, self.invalidDict, TestTemplates.plane_xy["name"]) 
    
    @params(System.createScalarGaussian, System.createPointSourceScalar, System.createUniformSourceScalar)
    def test_addPOScalarFields(self, func):
        ltot = len(TestTemplates.getPOSourceList())

        self.assertEqual(len(self.s.scalarfields), ltot)

        with self.assertRaises(InputPOError):
            func(self.s, self.invalidDict, TestTemplates.plane_xy["name"]) 

if __name__ == "__main__":
    import nose2
    nose2.main()
        
