"""!
@file
Script for testing the Sellmeier implementation for PyPO.
"""

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import unittest
import numpy as np

from PyPO.Sellmeier import BK7, FS

class Test_Sellmeier(unittest.TestCase):
    def test_BK7(self):
        lam = 2.5e-3
        ntest = 1.4860
        bk7 = BK7(lam)

        self.assertAlmostEqual(bk7.n, ntest, delta=1e-3)
    
    def test_FS(self):
        lam = 2.5e-3
        ntest = 1.4298
        fs = FS(lam)

        self.assertAlmostEqual(fs.n, ntest, delta=1e-3)

if __name__ == "__main__":
    import nose2
    nose2.main()
