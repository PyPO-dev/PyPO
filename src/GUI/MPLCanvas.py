import sys
import matplotlib
matplotlib.use('Qt5Agg')
sys.path.append('../../')
from examples.DRO_PSF_RT import ex_DRO

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, fig, parent=None, width=5, height=4, dpi=100):
        self.fig = fig
        super(MplCanvas, self).__init__(self.fig)

    def drawInCanvas(self, figure):
        super(MplCanvas, self).__init__(figure)
