from PyQt5 import QtWidgets as qtw
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import src.GUI.MPLCanvas as mplc



class PlotScreen(qtw.QWidget):
    def __init__(self, fig, parent = None):
        super().__init__(parent=parent)
        self.canvas = mplc.MplCanvas(fig)
        NavTB = NavigationToolbar(self.canvas, self)

        # self.setFixedHeight(400)

        layout = qtw.QVBoxLayout(self)
        layout.addWidget(NavTB)
        layout.addWidget(self.canvas)
if __name__ == "__main__":

    app = qtw.QApplication([])
    mainwindow = PlotScreen()
    mainwindow.show()
    app.exec_()
