import unittest
from PyPO.Checks import InputReflError, InputRTError, InputPOError

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates
##
# @file
# File containing tests for copy and remove methods for reflectors, fields, currents, frames and scalar fields.
class Test_SystemCopyAndRemove(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()
        self.s.setOverride(False) 

    @params(*TestTemplates.getAllReflectorList())
    def test_copyElement(self, element):
        self.s.copyElement(element["name"], element["name"] + "_copy")
        self.assertTrue(element["name"] + "_copy" in self.s.system)

        self._assert_copy(element["name"], element["name"] + "_copy")

    def test_removeElement(self):
        self.s.groupElements("testgroup", TestTemplates.paraboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"])
        self.s.removeElement(TestTemplates.paraboloid_man_xy["name"])
        
        self.assertFalse(TestTemplates.paraboloid_man_xy["name"] in self.s.groups["testgroup"]["members"])

        self.s.removeElement(TestTemplates.hyperboloid_man_xy["name"])
        self.s.removeElement(TestTemplates.ellipsoid_z_man_xy["name"])
        self.s.removeElement(TestTemplates.plane_xy["name"])

        self.assertFalse(TestTemplates.paraboloid_man_xy["name"] in self.s.system)
        self.assertFalse(TestTemplates.hyperboloid_man_xy["name"] in self.s.system)
        self.assertFalse(TestTemplates.ellipsoid_z_man_xy["name"] in self.s.system)
        self.assertFalse(TestTemplates.plane_xy["name"] in self.s.system)

    def test_copyGroup(self):
        self.s.groupElements("testgroup", TestTemplates.paraboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"])
        self.s.copyGroup("testgroup", "testgroup_copy")
        
        self._assert_copy("testgroup", "testgroup_copy", sdict="groups")

    @params(*TestTemplates.getFrameList())
    def test_removeFrame(self, frame):
        self.s.removeFrame(frame["name"])
        self.assertFalse(frame["name"] in self.s.frames)
    
    @params(*TestTemplates.getPOSourceList())
    def test_removeField(self, field):
        self.s.removeField(field["name"])
        self.assertFalse(field["name"] in self.s.fields)
    
    @params(*TestTemplates.getPOSourceList())
    def test_removeCurrent(self, current):
        self.s.removeCurrent(current["name"])
        self.assertFalse(current["name"] in self.s.currents)

    @params(*TestTemplates.getPOSourceList())
    def test_removeScalarField(self, field):
        self.s.removeScalarField(field["name"])
        self.assertFalse(field["name"] in self.s.scalarfields)

    def _assert_copy(self, name, name_copy, sdict="system"):
        if sdict == "system":
            sdict = self.s.system
        
        if sdict == "groups":
            sdict = self.s.groups
        
        self.assertTrue(name_copy in sdict)

        for (key1, item1), (key2, item2) in zip(sdict[name].items(), sdict[name_copy].items()):
            self.assertEqual(key1, key2)

            if isinstance(item1, int) or isinstance(item1, float) or isinstance(item1, bool) or isinstance(item1, str):
                self.assertEqual(id(item1), id(item2))
            
            else:
                self.assertNotEqual(id(item1), id(item2))


if __name__ == "__main__":
    import nose2
    nose2.main()
