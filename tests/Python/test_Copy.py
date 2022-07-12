import sys

sys.path.append('../../')

import unittest
import numpy as np

import src.Python.Copy as Copy

class TestCopy(unittest.TestCase):
   
    def test_copyGrid(self):
        xi = -100
        xe = 100
        
        yi = -100
        ye = 100
        
        num = 101
        
        x, y = np.mgrid[xi:xe:num*1j, yi:ye:num*1j]
        
        x0 = Copy.copyGrid(x)
        y0 = Copy.copyGrid(y)
        
        x += 666
        y += 999
        
        for xx0, yy0, xx, yy in zip(x0.ravel(), y0.ravel(), x.ravel(), y.ravel()):
            self.assertEqual(xx0 + 666, xx)
            self.assertEqual(yy0 + 999, yy)
        
if __name__ == "__main__":
    unittest.main()
