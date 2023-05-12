from PySide2.QtWidgets import QApplication, QMainWindow
                                    

import sys
sys.path.append('../')
sys.path.append('../../')

from src.GUI.ElementWidget import ElementWidget


class SystemWidget(ElementWidget):
    def __init__ (self, element, p=None ):
        super().__init__( element, p)


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,400)
    wid = ElementWidget("Parabola_0", window)
    
    window.show()
    app.exec_()