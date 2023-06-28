"""!
@file
Tests for checking if grids in PyPO are correct
"""

import unittest
import ctypes

from nose2.tools import params

try:
    from . import TestTemplates
except ImportError:
    import TestTemplates

import PyPO.BindRefl as refllibs
import PyPO.PyPOTypes as pypotypes

from PyPO.System import System

class Test_SystemGrids(unittest.TestCase):
    def setUp(self):
        self.s = TestTemplates.getSystemWithReflectors()

    def test_loadRefllib(self):
        lib = refllibs.loadRefllib()
        self.assertEqual(type(lib), ctypes.CDLL)
    
    @params(*TestTemplates.getAllSurfList())
    def test_generateGrids(self, element):
        grids = self.s.generateGrids(element["name"])
        self.assertEqual(type(grids), pypotypes.reflGrids)

if __name__ == "__main__":
    import nose2
    nose2.main()
