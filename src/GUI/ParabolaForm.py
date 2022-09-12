from PyQt5 import QtWidgets as qtw


class ParabolaFormLayout(qtw.QFormLayout):
    def __init__ (self):
        super().__init__()


        # self.defcombo = qtw.QFontComboBox()
        # self.defcombo.addItems("")

        nameEdit = qtw.QLineEdit()
        self.addRow("Name", nameEdit)

        coef = qtw.QLineEdit()
        self.addRow("Coefficient", coef)

        limxEdit = qtw.QLineEdit()
        self.addRow("X-Limit", limxEdit)

        limyEdit = qtw.QLineEdit()
        self.addRow("Y-Limit", limyEdit)


        
        # self.setStyleSheet("background: #D9D9D9;")



if __name__ == '__main__':

    app = qtw.QApplication([])
    widget = qtw.QWidget()
    widget.setLayout(ParabolaFormLayout())
    widget.setMaximumWidth(300)
    widget.show()
    app.exec_()