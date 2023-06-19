import unittest
import numpy as np

from PyPO.System import System

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

class Test_SystemSnapAndHome(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

        self.names = [TestTemplates.paraboloid_man_xy["name"], 
                TestTemplates.hyperboloid_man_xy["name"], TestTemplates.ellipsoid_z_man_xy["name"],
                TestTemplates.plane_xy["name"]]
        
        self.fr_names = [TestTemplates.TubeRTframe["name"], TestTemplates.GaussRTframe["name"]]

        self.s.groupElements("testgroup", *self.names)

    def test_snapObj(self):
        trans = np.array([-3, 42.42, 666])
        rot = np.array([359, -69, 69])

        for name in self.names:
            self.s.translateGrids(name, trans)
            self.s.rotateGrids(name, rot)
            self.s.snapObj(name, "test")

            self.assertFalse(id(self.s.system[name]["snapshots"]["test"]) == id(self.s.system[name]["transf"]))
            for tr, trs in zip(self.s.system[name]["snapshots"]["test"].ravel(), 
                                self.s.system[name]["transf"].ravel()):
                
                self.assertEqual(tr, trs)

        self.s.translateGrids("testgroup", trans, obj="group")
        self.s.rotateGrids("testgroup", rot, obj="group")
        
        self.s.snapObj("testgroup", "test", obj="group")


        for transfs, elem_name in zip(self.s.groups["testgroup"]["snapshots"]["test"], self.names):
            _transf = self.s.system[elem_name]["transf"]
            self.assertFalse(id(transfs) == id(_transf))

            for tr, trs in zip(transfs.ravel(), _transf.ravel()): 
                self.assertEqual(tr, trs)

        for fr_n in self.fr_names:
            self.s.translateGrids(fr_n, trans, obj="frame")
            self.s.rotateGrids(fr_n, rot, obj="frame")
            self.s.snapObj(fr_n, "test", obj="frame")

            self.assertFalse(id(self.s.frames[fr_n].snapshots["test"]) == id(self.s.frames[fr_n].transf))
            for tr, trs in zip(self.s.frames[fr_n].snapshots["test"].ravel(), 
                                self.s.frames[fr_n].transf.ravel()):
                
                self.assertEqual(tr, trs)

        for name in self.names:
            self.s.translateGrids(name, trans)
            self.s.rotateGrids(name, rot)
            self.s.revertToSnap(name, "test")

            for tr, trs in zip(self.s.system[name]["snapshots"]["test"].ravel(), 
                                self.s.system[name]["transf"].ravel()):
                
                self.assertEqual(tr, trs)

            self.s.deleteSnap(name, "test")
            self.assertFalse("test" in self.s.system[name]["snapshots"])

            self.s.homeReflector(name, rot=False)
            for tr, trs in zip([0, 0, 0, 1], self.s.system[name]["transf"][:,-1]):
                self.assertEqual(tr, trs)
            
            self.s.homeReflector(name, trans=False)
            for tr, trs in zip(np.eye(4).ravel(), self.s.system[name]["transf"].ravel()):
                self.assertEqual(tr, trs)

        self.s.translateGrids("testgroup", trans, obj="group")
        self.s.rotateGrids("testgroup", rot, obj="group")
        
        self.s.revertToSnap("testgroup", "test", obj="group")
        for transfs, elem_name in zip(self.s.groups["testgroup"]["snapshots"]["test"], self.names):
            _transf = self.s.system[elem_name]["transf"]
            self.assertFalse(id(transfs) == id(_transf))

            for tr, trs in zip(transfs.ravel(), _transf.ravel()): 
                self.assertEqual(tr, trs)

        self.s.deleteSnap("testgroup", "test", obj="group")
        self.assertFalse("test" in self.s.groups["testgroup"]["snapshots"])
        
        self.s.homeReflector("testgroup", obj="group", rot=False)
        for mem in self.s.groups["testgroup"]["members"]:
            for tr, trs in zip([0, 0, 0, 1], self.s.system[mem]["transf"][:,-1]):
                self.assertEqual(tr, trs)
        
        self.s.homeReflector("testgroup", obj="group", trans=False)
        for mem in self.s.groups["testgroup"]["members"]:
            for tr, trs in zip(np.eye(4).ravel(), self.s.system[mem]["transf"].ravel()):
                self.assertEqual(tr, trs)

        for fr_n in self.fr_names:
            self.s.translateGrids(fr_n, trans, obj="frame")
            self.s.rotateGrids(fr_n, rot, obj="frame")
            self.s.revertToSnap(fr_n, "test", obj="frame")

            self.assertFalse(id(self.s.frames[fr_n].snapshots["test"]) == id(self.s.frames[fr_n].transf))
            for tr, trs in zip(self.s.frames[fr_n].snapshots["test"].ravel(), 
                                self.s.frames[fr_n].transf.ravel()):
                
                self.assertEqual(tr, trs)
            
            self.s.deleteSnap(fr_n, "test", obj="frame")
            self.assertFalse("test" in self.s.frames[fr_n].snapshots)

    def test_homeReflector(self):
        trans = np.array([-3, 42.42, 666])
        rot = np.array([359, -69, 69])

        for name in self.names:
            self.s.translateGrids(name, trans)
            self.s.rotateGrids(name, rot)
    
            self.s.homeReflector(name, trans=False, rot=True)
            
            check_trans = self.s.copyObj(self.s.system[name]["transf"][:-1, -1])
            for ti, te in zip(self.s.system[name]["transf"][:-1, :-1].ravel(), np.eye(3).ravel()):
                self.assertEqual(ti, te)
    
            for ti, te in zip(self.s.system[name]["transf"][:-1, -1], check_trans):
                self.assertEqual(ti, te)
            
            self.s.rotateGrids(name, rot)

            self.s.homeReflector(name, trans=True, rot=False)

            check_rot = self.s.copyObj(self.s.system[name]["transf"][:-1, :-1]).ravel()
            for ti, te in zip(self.s.system[name]["transf"][:-1, :-1].ravel(), check_rot):
                self.assertEqual(ti, te)
            
            for ti, te in zip(self.s.system[name]["transf"][:-1, -1], np.zeros(3)):
                self.assertEqual(ti, te)

if __name__ == "__main__":
    import nose2
    nose2.main()
