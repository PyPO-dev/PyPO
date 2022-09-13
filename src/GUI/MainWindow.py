import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu
from MainWidget import MainWidget

class Window(QMainWindow):
    """Main Window."""
    def __init__(self, parent=None):
        """Initializer."""
        super().__init__(parent)
        self.setWindowTitle("POPPy")
        self._createMenuBar()
        
        self.centralWidget = MainWidget()
        self.setCentralWidget(self.centralWidget)




    def _createMenuBar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)

        ElementsMenu = QMenu("Elements", self)
        menuBar.addMenu(ElementsMenu)

        # ElementsMenu.addAction(self.AddElementAction)




    def AddElementAction(self):
        pass








if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Window()
    win.resize(1200,800)
    win.show()
    sys.exit(app.exec_())
