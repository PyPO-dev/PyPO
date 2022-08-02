from PyQt5 import QtWidgets as qtw



class ElementsWindow(qtw.QWidget):
    def __init__ (self, elements):
        super().__init__()

        

        elementsColumn = qtw.QVBoxLayout()
        for elem in elements:
            label = qtw.QLabel(elem)
            elementsColumn.addWidget(label)

        self.setStyleSheet("background: pink;")
        self.setLayout(elementsColumn)
        
