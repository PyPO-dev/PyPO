import sys
import matplotlib.pyplot as pt
sys.path.append('../../')

import unittest
import numpy as np

import src.Python.System as System

class TestParabola(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        print("\nTesting System") 
