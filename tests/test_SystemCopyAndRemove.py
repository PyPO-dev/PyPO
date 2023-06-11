import unittest
from PyPO.Checks import InputReflError, InputRTError, InputPOError

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

    def test_copyElement(self):
        for plane in TestTemplates.getPlaneList():
            self.s.copyElement(plane["name"], plane["name"] + "_copy")
            self.assertTrue(plane["name"] + "_copy" in self.s.system)

            self._assert_copy(plane["name"], plane["name"] + "_copy")
        
        for parabola in TestTemplates.getParaboloidList():
            self.s.copyElement(parabola["name"], parabola["name"] + "_copy")
            self.assertTrue(parabola["name"] + "_copy" in self.s.system)

            self._assert_copy(parabola["name"], parabola["name"] + "_copy")
        
        for hyperbola in TestTemplates.getHyperboloidList():
            self.s.copyElement(hyperbola["name"], hyperbola["name"] + "_copy")
            self.assertTrue(hyperbola["name"] + "_copy" in self.s.system)

            self._assert_copy(hyperbola["name"], hyperbola["name"] + "_copy")
        
        for ellipse in TestTemplates.getEllipsoidList():
            self.s.copyElement(ellipse["name"], ellipse["name"] + "_copy")
            self.assertTrue(ellipse["name"] + "_copy" in self.s.system)

            self._assert_copy(ellipse["name"], ellipse["name"] + "_copy")

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


    def test_removeFrame(self):
        for frame in TestTemplates.getFrameList():
            self.s.removeFrame(frame["name"])
            self.assertFalse(frame["name"] in self.s.frames)
    
    def test_removeField(self):
        for field in TestTemplates.getPOSourceList():
            self.s.removeField(field["name"])
            self.assertFalse(field["name"] in self.s.fields)
    
    def test_removeCurrent(self):
        for current in TestTemplates.getPOSourceList():
            self.s.removeCurrent(current["name"])
            self.assertFalse(current["name"] in self.s.currents)

    def test_removeScalarField(self):
        for field in TestTemplates.getPOSourceList():
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
    unittest.main()
        


