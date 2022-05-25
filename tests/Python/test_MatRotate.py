import sys

sys.path.append('../../')

import unittest
import numpy as np

import src.Python.MatRotate as MatRotate

class TestMatRotate(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\nTesting MatRotate")
    
    def setUp(self):
        self.theta1 = np.radians(np.array([0,0,180]))
        self.theta2 = np.radians(np.array([0,180,0]))
        self.theta3 = np.radians(np.array([90,90,-90]))
        
        self.point1 = np.array([1,0,0])
        self.point2 = np.array([0,0,1])
        self.point3 = np.array([0,1,0])
    
    def tearDown(self):
        pass
    
    def test_MatRotate(self):
        result, mat = MatRotate.MatRotate(theta=self.theta1, points=self.point1, matOut=True)
        self.assertEqual(result[0], -1)
        
        result_inv = np.dot(mat.T, result)
        
        for x, y in zip(self.point1, result_inv):
            self.assertAlmostEqual(x, y) 

        result, mat = MatRotate.MatRotate(theta=self.theta2, points=self.point2, matOut=True)
        self.assertEqual(result[2], -1)
        
        result_inv = np.dot(mat.T, result)
        
        for x, y in zip(self.point2, result_inv):
            self.assertAlmostEqual(x, y) 
        
        result, mat = MatRotate.MatRotate(theta=self.theta3, points=self.point3, matOut=True)
        self.assertEqual(result[1], -1)
        
        result_inv = np.dot(mat.T, result)
        
        for x, y in zip(self.point3, result_inv):
            self.assertAlmostEqual(x, y) 
        
        result_tot_test = result
        mat_to_test = mat
        
        result, mat = MatRotate.MatRotate(theta=self.theta3, points=result, matOut=True)
        self.assertEqual(result[1], 1)
        
        result_inv = np.dot(mat.T, result)
        
        for x, y in zip(result_tot_test, result_inv):
            self.assertAlmostEqual(x, y) 
            
        result_inv = np.dot(mat_to_test.T, result_tot_test)
            
        for x, y in zip(self.point3, result_inv):
            self.assertAlmostEqual(x, y)

if __name__ == "__main__":
    unittest.main()
