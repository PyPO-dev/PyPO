import unittest
from PyPO.System import System
# import TestTemplates 
try:
    from . import TestTemplates
except ImportError: 
    import TestTemplates

from numpy import ndarray 
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes



stm = TestTemplates.getSystemWithReflectors()

class Test_SystemOps(unittest.TestCase):

    # def test_plotBeamCut(self):
    #     # stm.createGaussian(TestTemplates.GPOfield, 'testPlane_xy')

    #     # print(stm.fields,'!!!!!!!!')

    #     fig, ax = stm.plotBeamCut(TestTemplates.GPOfield['name'], 'Ex', ret = True)

    #     self.assertEqual(type(fig), Figure)
    #     self.assertEqual(type(ax), Axes)

    def test_plotBeam2D(self):

        fig1, ax1 = stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret = True, amp_only=True)
        fig2, ax2 = stm.plotBeam2D(TestTemplates.GPOfield['name'], 'Ex', ret = True)

        self.assertEqual(type(fig1), Figure)
        self.assertEqual(type(ax1), Axes)
        self.assertEqual(type(fig2), Figure)
        self.assertEqual(type(ax2), ndarray)
        

if __name__ == '__main__':
    unittest.main()


