import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from PyPO.System import System

class Test_MergeBeams(unittest.TestCase):
    def test_mergeBeams_uv(self):
        for i in range(1):
            self._generate_uniform_uv()
            self.s.mergeBeams("source1", "source2", merged_name="combined")
            for i in range(6):
                field_test = self.s.fields["source1"][i] + self.s.fields["source2"][i]
                for valc, valt in zip(self.s.fields["combined"][i].ravel(), field_test.ravel()):
                    self.assertAlmostEqual(valc, valt)
            del self.s
    
    def test_mergeBeams_xy(self):
        for i in range(1):
            self._generate_uniform_xy()
            self.s.mergeBeams("source1", "source2", merged_name="combined")
            for i in range(6):
                field_test = self.s.fields["source1"][i] + self.s.fields["source2"][i]
                for valc, valt in zip(self.s.fields["combined"][i].ravel(), field_test.ravel()):
                    self.assertAlmostEqual(valc, valt)
            del self.s

    def _generate_uniform_uv(self):
        self.s = System(override=False, verbose=False)

        xi = random.uniform(0, 10)
        xo = random.uniform(xi, 100)
        
        ring = {
                "name"      : "ring",
                "gmode"     : "uv",
                "lims_u"    : np.array([xi, xo]),
                "lims_v"    : np.array([0, 360]),
                "gridsize"  : np.array([101, 101])
                }

        UDict1 = {
                "name"      : "source1",
                "lam"       : 1
                }
        
        UDict2 = {
                "name"      : "source2",
                "lam"       : 1
                }

        self.s.addPlane(ring)
        self.s.createUniformSource(UDict1, "ring")
        self.s.createUniformSource(UDict2, "ring")
    
    def _generate_uniform_xy(self):
        self.s = System(override=False, verbose=False)

        xi = random.uniform(0, -100)
        xo = random.uniform(0, 100)
        
        yi = random.uniform(0, -100)
        yo = random.uniform(0, 100)
        
        ring = {
                "name"      : "ring",
                "gmode"     : "xy",
                "lims_x"    : np.array([xi, xo]),
                "lims_y"    : np.array([yi, yo]),
                "gridsize"  : np.array([101, 101])
                }

        UDict1 = {
                "name"      : "source1",
                "lam"       : 1
                }
        
        UDict2 = {
                "name"      : "source2",
                "lam"       : 1
                }

        self.s.addPlane(ring)
        self.s.createUniformSource(UDict1, "ring")
        self.s.createUniformSource(UDict2, "ring")

if __name__ == "__main__":
    unittest.main()

