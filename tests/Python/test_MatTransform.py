import sys
import random

sys.path.append('../../')

import unittest
import numpy as np

import src.POPPy.MatTransform as MatTransform

class TestMatTransform(unittest.TestCase):
    """
    Test for rotation and translation functions.
    Test 1: apply 5 random rotations and 5 random translations.
    Apply inverse to see if we recover identity
    """

    @classmethod
    def setUpClass(cls):
        print("\nTesting MatTransform")
    
    def setUp(self):
        self.theta1 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 180
        self.theta2 = (np.array([random.random(),random.random(),random.random()])  - 0.5) * 180
        self.theta3 = (np.array([random.random(),random.random(),random.random()])  - 0.5) * 180
        self.theta4 = (np.array([random.random(),random.random(),random.random()])  - 0.5) * 180
        self.theta5 = (np.array([random.random(),random.random(),random.random()])  - 0.5) * 180

        self.origin1 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.origin2 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.origin3 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.origin4 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.origin5 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        
        self.point1 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.point2 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.point3 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.point4 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
        self.point5 = (np.array([random.random(),random.random(),random.random()]) - 0.5) * 10
    
    def tearDown(self):
        pass
    
    def test_MatTransform(self):
        matAppend = np.eye(4)
        matAccum = np.eye(4)

        matAccum = MatTransform.MatRotate(self.theta1, matAppend, self.origin1)
        matAccum = MatTransform.MatTranslate(self.point1, matAccum)

        matAccum = MatTransform.MatRotate(self.theta2, matAccum, self.origin2)
        matAccum = MatTransform.MatTranslate(self.point2, matAccum)

        matAccum = MatTransform.MatRotate(self.theta3, matAccum, self.origin3)
        matAccum = MatTransform.MatTranslate(self.point3, matAccum)

        matAccum = MatTransform.MatRotate(self.theta4, matAccum, self.origin4)
        matAccum = MatTransform.MatTranslate(self.point4, matAccum)

        matAccum = MatTransform.MatRotate(self.theta5, matAccum, self.origin5)
        matAccum = MatTransform.MatTranslate(self.point5, matAccum)

        matInv = MatTransform.InvertMat(matAccum)

        matRes = np.matmul(matInv, matAccum)

        for x, y in zip(matRes.ravel(), matAppend.ravel()):
            self.assertAlmostEqual(x, y)

if __name__ == "__main__":
    unittest.main()
