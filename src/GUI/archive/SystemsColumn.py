from PyQt5.QtWidgets import QWidget, QApplication
# from PyQt5 import QtCore
# from PyQt5.QtGui import QFont, QIcon


from GUI.archive.ElementsColumn import ElementsWindow
from src.GUI.archive.ElementWidget_Systems import SystemWidget

class SystemsWindow(QWidget):
    def __init__ (self, elements):
        super().__init__()
        pushDown = 0
        for elem in elements:
            label = SystemWidget(elem, p=self)
            label.move(0,pushDown)
            pushDown += 40
            # ColLayout.addWidget(label)

        # verticalSpacer = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Maximum, qtw.QSizePolicy.Expanding)         
        # ColLayout.addItem(verticalSpacer)

        self.setMaximumHeight(500)

        
   
        
if __name__ == '__main__':

    app = QApplication()
    mw = ElementsWindow(["lala", "lolo"])
    mw.resize(400,300)
    mw.show()
    app.exec_()