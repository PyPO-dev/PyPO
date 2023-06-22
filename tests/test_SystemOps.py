import os
import shutil

import unittest

from PyPO.System import System

class Test_SystemOps(unittest.TestCase):
    def setUp(self):
        self.s0 = System(verbose=False)
        self.s1 = System(verbose=False)

        self.params = [["system", 2], 
                        ["fields", 3],
                        ["currents", 4],
                        ["frames", 5],
                        ["scalarfields", 6],
                        ["groups", 7]]
        
        self.content = ["system.pys", "groups.pys", "frames.pys", "fields.pys", "currents.pys",
                        "scalarfields.pys"]

        for i in range(3):
            getattr(self.s0, self.params[i][0])[self.params[i][0]] = self.params[i][1]
            getattr(self.s1, self.params[3+i][0])[self.params[3+i][0]] = self.params[3+i][1]
   
    def test_mergeSystem(self):
        self.s0.mergeSystem(self.s1)
        for par in self.params:
            self.assertEqual(getattr(self.s0, par[0])[par[0]], par[1])

    def test_saveSystem(self):
        self.s0.saveSystem("s0")
        self.s1.saveSystem("s1")
        
        self.assertTrue(os.path.exists(os.path.join(self.s0.savePathSystems, "s0.pyposystem"))) 
        self.assertTrue(os.path.exists(os.path.join(self.s1.savePathSystems, "s1.pyposystem")))

        os.remove(os.path.join(self.s0.savePathSystems, "s0.pyposystem"))
        os.remove(os.path.join(self.s1.savePathSystems, "s1.pyposystem"))

        self.assertFalse(os.path.exists(os.path.join(self.s0.savePathSystems, "s0.pyposystem"))) 
        self.assertFalse(os.path.exists(os.path.join(self.s1.savePathSystems, "s1.pyposystem")))

    def test_loadSystem(self):
        self.s0.mergeSystem(self.s1)
        self.s0.saveSystem("s0")
       
        self.s1.loadSystem("s0")
        for par in self.params:
            self.assertEqual(getattr(self.s1, par[0])[par[0]], par[1])
        
        os.remove(os.path.join(self.s1.savePathSystems, "s0.pyposystem"))
        self.assertFalse(os.path.exists(os.path.join(self.s0.savePathSystems, "s0.pyposystem"))) 

    def test_deleteSystem(self):
        del self.s0

if __name__ == "__main__":
    import nose2
    nose2.main()
