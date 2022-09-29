from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore
from PyQt5.QtGui import QFont, QIcon


from ElementWidget import ElementWidget

class ElementsWindow(qtw.QWidget):
    def __init__ (self, elements):
        super().__init__()

        ColLayout = qtw.QVBoxLayout()
        pushDown = 0
        for elem in elements:
            label = ElementWidget(self, elem)
            label.move(0,pushDown)
            pushDown += 40
            # ColLayout.addWidget(label)

        # verticalSpacer = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Maximum, qtw.QSizePolicy.Expanding)         
        # ColLayout.addItem(verticalSpacer)


        # self.setStyleSheet("background: #5A0168; color:white")
        # self.setLayout(ColLayout)
        self.setMaximumHeight(500)
   

        
if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = ElementsWindow(["lala", "lolo"])
    mw.resize(400,300)
    mw.show()
    app.exec_()