from PyQt5 import QtWidgets as qtw



class ElementWidget(qtw.QWidget):
    def __init__ (self, element):
        super().__init__()

        self.layout = qtw.QHBoxLayout()

        self.nameTag = qtw.QPushButton(element.Name)
        self.nameTag.activated.connect(self.setForm)

        self.layout.addWidget(self.nameLabel)
        self.optionsBTN = qtw.QPushButton()





    def setForm(self):
        pass
        

        

if __name__ == "__main__":
    app = qtw.QApplication([])
    mw = ElementWidget()
    mw.show()
    app.exec_()