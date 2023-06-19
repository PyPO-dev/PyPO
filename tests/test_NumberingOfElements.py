import unittest
from PyPO.System import System
from PyPO.Checks import InputReflError, InputRTError, InputPOError

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

class Test_SystemDictsAndAddElement(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False) 
        
        self.funcSelect = {
                0 : self.s.addParabola,
                1 : self.s.addHyperbola,
                2 : self.s.addEllipse,
                3 : self.s.addPlane
                }
    
    @params(*TestTemplates.getAllReflectorList())
    def test_namingReflector(self, element):
        for i in range(2):
            self.funcSelect[self.s.system[element["name"]]["type"]](element)
            self.assertTrue(element['name'] + f"_{i+1}" in self.s.system)

    def test_namingGroup(self):
        self.s.groupElements('g')
        self.s.groupElements('g')
        self.s.groupElements('g')
        self.assertTrue("g_1" in self.s.groups)
        self.assertTrue("g_2" in self.s.groups)

    def test_namingFrames(self):
        for i in range(2):
            self.s.createTubeFrame(TestTemplates.TubeRTframe)
            self.assertTrue(TestTemplates.TubeRTframe["name"] + f"_{i+1}" in self.s.frames)
        
            self.s.createGRTFrame(TestTemplates.GaussRTframe)
            self.assertTrue(TestTemplates.GaussRTframe["name"] + f"_{i+1}" in self.s.frames)

    @params(*TestTemplates.getPlaneList())
    def test_addPOFields(self, plane):
        if plane["gmode"] == "AoE":
            return

        for i in range(2):
            self.s.createPointSource(TestTemplates.PS_Ufield, plane["name"])
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.fields)
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.currents)
            
            self.s.createGaussian(TestTemplates.GPOfield, plane["name"])
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.fields)
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.currents)
        
            self.s.createPointSourceScalar(TestTemplates.PS_Ufield, plane["name"])
            self.assertTrue(TestTemplates.PS_Ufield["name"] + f"_{i+1}" in self.s.scalarfields)
            
            self.s.createScalarGaussian(TestTemplates.GPOfield, plane["name"])
            self.assertTrue(TestTemplates.GPOfield["name"] + f"_{i+1}" in self.s.scalarfields)

if __name__ == "__main__":
    import nose2
    nose2.main()
