from PySide2.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QFormLayout , QLabel, QTabBar, QTabWidget
import numpy as np



class PlotTabWidget(QTabBar):
    def __init__ (self):
        super().__init__()
        w = QWidget()
        layout = QFormLayout()
        layout.addRow(QLabel("Info"), QLineEdit())
        w.setLayout(layout)

        self.addTab("TTT1")


if __name__ == '__main__':
    app = QApplication([])
    mw = PlotTabWidget()
    mw.resize(400,300)
    mw.show()
    app.exec_()