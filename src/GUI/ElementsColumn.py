from PyQt5 import QtWidgets as qtw



class ElementsWindow(qtw.QWidget):
    def __init__ (self, elements, btn):
        super().__init__()

        elementsColumn = qtw.QVBoxLayout()
        elementsColumn.addWidget(btn)
        for elem in elements:
            label = qtw.QLabel(elem)
            label.setFixedHeight(32)
            elementsColumn.addWidget(label)

        verticalSpacer = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Expanding)         
        elementsColumn.addItem(verticalSpacer)

        # self.setStyleSheet("background: #5A0168; color:white")
        self.setLayout(elementsColumn)
        
if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = ElementsWindow(["lala", "lolo"], qtw.QPushButton("add element"))
    mw.show()
    app.exec_()