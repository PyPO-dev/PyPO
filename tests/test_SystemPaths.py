import sys
import os
import random
import shutil

import unittest
import numpy as np
from pathlib import Path

from PyPO.System import System
##
# @file
# 
# Script for testing the path setting functionality in PyPO.

class Test_SystemPaths(unittest.TestCase):
    def setUp(self):
        self.s = System(verbose=False)
        self.filePath = Path(__file__).parents[0]
        self.path_test = os.path.join(self.filePath, "test/")
        self.path_app = "append/"

    def test_setCustomBeamPath(self):
        self.s.setCustomBeamPath(self.path_test)
        self.assertEqual(self.path_test, self.s.customBeamPath)

        self.s.setCustomBeamPath(self.path_app, append=True)
        self.assertEqual(os.path.join(self.path_test, self.path_app), self.s.customBeamPath)

    def test_setSavePath(self):
        funcs = ["setSavePath", "setSavePathSystems"]

        for func in funcs:
            getattr(self.s, func)(self.path_test)
            path = getattr(self.s, "s" + func.split("setS")[1]) 
            self.assertEqual(self.path_test, path)
            
            getattr(self.s, func)(self.path_app, append=True)
            path = getattr(self.s, "s" + func.split("setS")[1]) 
            self.assertEqual(os.path.join(self.path_test, self.path_app), path)

            self.assertTrue(os.path.exists(os.path.join(self.path_test, self.path_app)))
            shutil.rmtree(self.path_test)
            self.assertFalse(os.path.exists(os.path.join(self.path_test, self.path_app)))

if __name__ == "__main__":
    unittest.main()
