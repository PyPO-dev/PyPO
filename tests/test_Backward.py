import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from src.PyPO.System import System

class Test_Backwards(unittest.TestCase):
    def test_Backwards(self):
        s = System(override=True, verbose=False)
        
        source = {
                "name"      : "source",
                "gmode"     : "xy",
                "lims_x"    : np.array([-0.01, 0.01]),
                "lims_y"    : np.array([-0.01, 0.01]),
                "gridsize"  : np.array([31, 31])
                }
        
        plane_up = {
                "name"      : "plane_up",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 100]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([301, 301]),
                "flip"      : True
                }

        plane_down = {
                "name"      : "plane_down",
                "gmode"     : "uv",
                "lims_u"    : np.array([0, 0.1]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }
        
        s.addPlane(source)
        s.addPlane(plane_up)
        s.addPlane(plane_down)

        for i in range(10):
            rPS, phase_check = self._random_PS()
            s.createPointSource(rPS, "source")

            s.translateGrids("plane_up", np.array([0, 0, 100]))
            
            runPODict = {
                    "t_name"    : "plane_up",
                    "s_current" : "PS_source",
                    "epsilon"   : 10,
                    "exp"       : "fwd",
                    "mode"      : "JMEH",
                    "name_JM"   : "JM_up",
                    "name_EH"   : "EH_up"
                    }
            
            runPODict_bwd = {
                    "t_name"    : "plane_down",
                    "s_current" : "JM_up",
                    "epsilon"   : 10,
                    "exp"       : "bwd",
                    "mode"      : "JMEH",
                    "name_JM"   : "JM_down",
                    "name_EH"   : "EH_down"
                    }

            s.runPO(runPODict)
            s.runPO(runPODict_bwd)

            phase_Ex = np.mean(np.angle(s.fields["EH_down"].Ex))
            self.assertAlmostEqual(phase_Ex, phase_check, delta=1e-2)

    def _random_PS(self):
        phase_PS = random.uniform(-np.pi, np.pi)

        PSDict = {
                "name"      : "PS_source",
                "lam"       : 1,
                "E0"        : 1,
                "phase"     : phase_PS,
                "pol"       : np.array([1,0,0])
                }

        return PSDict, phase_PS

if __name__ == "__main__":
    unittest.main()
