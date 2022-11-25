import numpy as np
import sys
import unittest
import numpy as np

import src.POPPy.Checks as Checks

class TestChecks(unittest.TestCase):
   
    def test_ElemDict(self):
        """
        This test will check if all errors are caught by the checker in reflDicts.
        First, we test a parabola. Then hyperbola, ellipse and plane.
        """

        par_man_t = {
                "type"      : 0,
                "name"      : "par_man_t",
                "pmode"     : "manual",
                "coeffs"    : np.array([1, 1]),
                "gmode"     : "xy",
                "lims_x"    : np.array([-1, 1]),
                "lims_y"    : np.array([-1, 1]),
                "gridsize"  : np.array([101, 101])
                }

        par_foc_t = {
                "type"      : 0,
                "name"      : "par_foc_t",
                "pmode"     : "focus",
                "vertex"    : np.zeros(3),
                "focus_1"   : np.array([0, 0, 1]),
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        # Test if correct dictionary returns 0
        succes = Checks.check_ElemDict(par_man_t)
        self.assertEqual(succes, 0)
        
        # Now, start rasing and catching exceptions
        # First, spelling error in pmode
        par_man_t["pmode"] = "manuel" 
        self.assertRaises(Checks.InputReflError, Checks.check_ElemDict, par_man_t)

        # Remove pmode
        del par_man_t["pmode"]
        self.assertRaises(Checks.InputReflError, Checks.check_ElemDict, par_man_t)
        par_man_t["pmode"] = "manual"

        # Adjust length of coeffs field and try illegal values
        par_man_t["coeffs"] = np.ones(3)
        self.assertRaises(Checks.InputReflError, Checks.check_ElemDict, par_man_t)
        #par_man_t["coeffs"] = -np.ones(2)

        # Remove coeffs
        del par_man_t["coeffs"]
        self.assertRaises(Checks.InputReflError, Checks.check_ElemDict, par_man_t)

        #### TODO
        # NEED TO INCLUDE ALL POSSIBLE FAILURES

if __name__ == "__main__":
    unittest.main()
