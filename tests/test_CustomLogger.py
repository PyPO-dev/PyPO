import unittest
import logging

from PyPO.CustomLogger import addLoggingLevel, CustomFormatter, CustomLogger, GUILogger, CustomGUIFormatter, CustomGUILogger

class TestCustomLogger(unittest.TestCase):
    def test_addLoggingLevel(self):
        addLoggingLevel("TEST", logging.INFO)
        self.assertTrue(logging.TEST)
        self.assertEqual(logging.TEST, logging.INFO)

    def test_CustomFormatter(self):
        test_cf = CustomFormatter()
        self.assertEqual(type(test_cf), CustomFormatter)
    
    def test_CustomGUIFormatter(self):
        test_cf = CustomGUIFormatter()
        self.assertEqual(type(test_cf), CustomGUIFormatter)

    def test_CustomLogger(self):
        test_cl = CustomLogger()
        self.assertEqual(type(test_cl), CustomLogger)

        test_clo = test_cl.getCustomLogger(stdout="test")
        self.assertEqual(type(test_clo), logging.Logger)
    
    def test_CustomGUILogger(self):
        test_cl = CustomGUILogger()
        self.assertEqual(type(test_cl), CustomGUILogger)

        test_clo = test_cl.getCustomGUILogger(TextEditWidget="test")
        self.assertEqual(type(test_clo), logging.Logger)

    def test_GUILogger(self):
        test_gl = GUILogger()
        self.assertEqual(type(test_gl), GUILogger)

if __name__ == "__main__":
    unittest.main()
