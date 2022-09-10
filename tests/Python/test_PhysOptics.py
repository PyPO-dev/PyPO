import sys
import os

sys.path.append('../../')

import unittest
import numpy as np

from src.Python.PhysOptics import PhysOptics

class TestPhysOptics(unittest.TestCase): 
    @classmethod
    def setUpClass(cls):
        print("\nTesting PO interface")
        datadir = './datadir/'
        exists_datadir = os.path.isdir(datadir)

        if not exists_datadir:
            os.makedirs('./datadir/')
    
    def setUp(self):
        test_path = './datadir/'
        self.PO = PhysOptics(k=1, numThreads=1, thres=-1, cpp_path=test_path)
        self.gridsize = [501,501]
        
    def TearDown(self):
        pass
    
    def test_writeInput(self):
        toWrite = np.eye(self.gridsize[0])
        self.t1 = 'PO_1.txt'
        self.PO.writeInput(self.t1, toWrite)
        
        toTest = np.loadtxt('./datadir/input/PO_1.txt')
        toTest = toTest.reshape(self.gridsize)
        
        self.assertEqual(np.sum(np.diag(toWrite)), np.sum(np.diag(toTest)))
        
if __name__ == "__main__":
    unittest.main()
