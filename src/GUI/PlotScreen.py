from PyQt5 import QtWidgets as qtw
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
import MPLCanvas as mplc



class PlotScreen(qtw.QWidget):
    def __init__(self):
        super().__init__()
        canvas = mplc.MplCanvas(self)
        NavTB = NavigationToolbar(canvas, self)

        layout = qtw.QVBoxLayout()
        layout.addWidget(NavTB)

        self.setLayout(layout)


if __name__ == "__main__":

    app = qtw.QApplication([])
    mainwindow = PlotScreen()
    mainwindow.show()
    app.exec_()
