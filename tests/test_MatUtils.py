import unittest
import numpy as np

from PyPO.MatUtils import findConnectedSubsets
##
# @file
# 
# Script for testing the matrix utilities of PyPO.

class Test_MatUtils(unittest.TestCase):
    def test_findConnectedSubsets(self): 
        mat = np.zeros((6,6))
        mat[1,2] = 1
        mat[0,2] = 1
        mat[2,2] = 1
        mat[1,3] = 1
        mat[1,1] = 1

        lims_0_check = [0, 1, 2]
        lims_1_check = [1, 2, 3]

        mat[5,5] = 1
        mat[5,4] = 1
        mat[5,3] = 1
        mat[5,2] = 1
        
        component = 1
        idx_start = [1,2]

        x, y = findConnectedSubsets(mat, component, idx_start)
        for l0, xx in zip(lims_0_check, x):
            self.assertEqual(l0, xx)
        for l1, yy in zip(lims_1_check, y):
            self.assertEqual(l1, yy)
        
if __name__ == "__main__":
    import nose2
    nose2.main()
