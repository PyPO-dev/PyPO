from PyQt5 import QtWidgets as qtw
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import src.GUI.MPLCanvas as mplc



class PlotScreen(qtw.QWidget):
    def __init__(self, fig):
        super().__init__()
        self.canvas = mplc.MplCanvas(fig)
        NavTB = NavigationToolbar(self.canvas, self)

        layout = qtw.QVBoxLayout()
        layout.addWidget(NavTB)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

if __name__ == "__main__":

    app = qtw.QApplication([])
    mainwindow = PlotScreen()
    mainwindow.show()
    app.exec_()
