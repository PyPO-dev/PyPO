import os
import shutil
import numpy as np

import unittest
from nose2.tools import params

from PyPO.System import System
from PyPO.PyPOTypes import fields

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

    def test_printSystem(self):
        test_str = self.s0.__str__()
        self.assertTrue(isinstance(test_str, str))

    @params("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") 
    def test_compToFields(self, comp):
        test_arr = np.zeros((3,3))
        out = self.s0._compToFields(comp, test_arr)
        
        self.assertTrue(isinstance(out, fields))

    @params("m", "cm", "mm", "um", "nm", "deg", "am", "as") 
    def test_units(self, unit):
        out = self.s0._units(unit)
        self.assertEqual(out[0], unit)
        self.assertEqual(len(out), 2)
        
if __name__ == "__main__":
    import nose2
    nose2.main()
