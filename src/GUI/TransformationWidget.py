from PyQt5 import QtWidgets as qtw
import numpy as np



class TransformationWidget(qtw.QWidget):
    def __init__ (self):
        super().__init__()

        self.layout = qtw.QHBoxLayout()
        self.layout.addLayout(self.makeForm())
        self.layout.addLayout(self.makeButtons())

        self.setLayout(self.layout)

    def setElement(self, element):
        self.element = element


    def makeForm(self):
        self.trans1, self.trans2, self.trans3 = qtw.QLineEdit(), qtw.QLineEdit(), qtw.QLineEdit()
        self.rot1, self.rot2, self.rot3 = qtw.QLineEdit(), qtw.QLineEdit(), qtw.QLineEdit()
        transLayout = qtw.QHBoxLayout()
        transLayout.addWidget(self.trans1)
        transLayout.addWidget(self.trans2)
        transLayout.addWidget(self.trans3)
        transWidget = qtw.QWidget()
        transWidget.setLayout(transLayout)
        rotLayout = qtw.QHBoxLayout()
        rotLayout.addWidget(self.rot1)
        rotLayout.addWidget(self.rot2)
        rotLayout.addWidget(self.rot3)
        rotWidget = qtw.QWidget()
        rotWidget.setLayout(rotLayout)
        
        form = qtw.QFormLayout()
        form.addRow(qtw.QLabel("Translation"), transWidget)
        form.addRow(qtw.QLabel("Rotation"), rotWidget)
        return form

    def makeButtons(self):
        self.buttonsLayout = qtw.QVBoxLayout()

        self.translateBtn = qtw.QPushButton("Translate")
        self.translateBtn.clicked.connect(self.translate)
        self.buttonsLayout.addWidget(self.translateBtn)

        self.rotateBtn = qtw.QPushButton("Rotate")
        self.rotateBtn.clicked.connect(self.rotate)
        self.buttonsLayout.addWidget(self.rotateBtn)
        return self.buttonsLayout

    def translate(self):
        x = float(self.trans1.text())
        y = float(self.trans2.text())
        z = float(self.trans3.text())

        trans = np.array([x, y, z])

        self.element.translateGrid(trans)

    def rotate(self):
        x = float(self.rot1.text())
        y = float(self.rot2.text())
        z = float(self.rot3.text())

        rots = np.array([x, y, z])

        self.element.rotateGrid(rots)



if __name__ == '__main__':

    app = qtw.QApplication([])
    mw = TransformationWidget()
    mw.show()
    app.exec_()