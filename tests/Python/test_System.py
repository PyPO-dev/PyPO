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
        
    def setUp(self):
        self.system = System.System()
    
    def test_addParabola(self):
        """
        Use the ASTE primary to test the parabola function.
        """
        
        a = 118.32159566199232
        b = 118.32159566199232
        
        foc = np.array([0, 0, 3500])
        ver = np.zeros(3)
        
        
        
        
