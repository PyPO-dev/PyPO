from PySide2.QtWidgets import  QWidget, QVBoxLayout
from matplotlib import use
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar

##
# @file Defines a matplotlib plot widget that can be placed in the GUI.
use('Qt5Agg')
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, fig):
        super(MplCanvas, self).__init__(fig)

##
# Generate plot widget.
class PlotScreen(QWidget):
    def __init__(self, fig, parent = None):
        super().__init__(parent=parent)
        canvas = MplCanvas(fig)
        NavTB = NavigationToolbar(canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(NavTB)
        layout.addWidget(canvas)
