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
        # self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def drawInCanvas(self, figure):
        print("??????????")
        super(MplCanvas, self).__init__(figure)


# class MainWindow(QtWidgets.QMainWindow):

#     def __init__(self, *args, **kwargs):
#         super(MainWindow, self).__init__(*args, **kwargs)

#         canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        
#         # sc.axes.plot([0,1,2,3,4], [10,1,20,3,40])
#         # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
#         toolbar = NavigationToolbar(canvas, self)

#         layout = QtWidgets.QVBoxLayout()
#         layout.addWidget(toolbar)
#         layout.addWidget(canvas)
        

#         # Create a placeholder widget to hold our toolbar and canvas.
#         widget = QtWidgets.QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)

#         self.show()

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     w = MainWindow()
#     app.exec_()