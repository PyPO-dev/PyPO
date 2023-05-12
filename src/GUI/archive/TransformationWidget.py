from PySide2.QtWidgets import QApplication, QWidget, QHBoxLayout, QLineEdit, QPushButton, QFormLayout , QLabel, QVBoxLayout, QComboBox, QGridLayout
from PySide2.QtGui import QRegExpValidator
from PySide2.QtCore import QRegExp, Qt

import numpy as np



class TransformationWidget(QWidget):
    def __init__ (self, elem, applyTransformation):
        super().__init__()

        self.element = elem
        self.applyTransformation = applyTransformation
        self.transformationType = None

        self.layout = QGridLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.makeStart()
        self.makeButtons()

        self.setLayout(self.layout)

    def makeStart(self):
        TransformedElement = QWidget()
        elementLayout = QHBoxLayout()
        elementLayout.addWidget(QLabel("Element:  "))
        elementLayout.addWidget(QLabel(self.element))
        TransformedElement.setLayout(elementLayout)

        self.transtype = QComboBox()
        self.transtype.addItems(["--Select Item--", "Translation", "Rotation"])
        self.transtype.activated.connect(self.transtypeChanged)

        self.layout.addWidget(TransformedElement,0,0)
        self.layout.addWidget(self.transtype,1,0)
        placeholderForm = QWidget()
        placeholderForm.setFixedSize(300,150)
        self.layout.addWidget(placeholderForm,2,0)
        
    def transtypeChanged(self):
        if self.transtype.currentIndex() == 1:
            self.makeForm("trans")
            self.transformationType = "trans"
        elif self.transtype.currentIndex() == 2:
            self.makeForm("rot")
            self.transformationType = "rot"

    def makeForm(self, type):
        if hasattr(self, "form"):
            self.form.setParent(None)

        self.form = QWidget()
        formLayout = QFormLayout()

        self.trans = [QLineEdit(), QLineEdit(), QLineEdit()]
        transLayout = QHBoxLayout()
        for i in self.trans:
            transLayout.addWidget(i)
        transWidget = QWidget()
        # transWidget.setFixedSize(150, 75)
        transWidget.setLayout(transLayout)
    
        self.rot = [QLineEdit(), QLineEdit(), QLineEdit()]
        rotLayout = QHBoxLayout()
        for i in self.rot:
            rotLayout.addWidget(i)
        rotWidget = QWidget()
        # rotWidget.setFixedSize(150, 75)
        rotWidget.setLayout(rotLayout)

        self.rotC = [QLineEdit(), QLineEdit(), QLineEdit()]
        rotCLayout = QHBoxLayout()
        for i in self.rotC:
            rotCLayout.addWidget(i)
        rotCWidget = QWidget()
        # rotCWidget.setFixedSize(150, 75)
        rotCWidget.setLayout(rotCLayout)

        if type == "trans":
            formLayout.addRow(QLabel("Translation Vector"), transWidget)
        elif type == "rot":
            formLayout.addRow(QLabel("Rotation angels (Deg)"), rotWidget)
            formLayout.addRow(QLabel("Center of rotation"), rotCWidget)
        
        self.form.setLayout(formLayout)

        self.layout.addWidget(self.form,2,0)

        ### input hints and validations
        hints = ["x", "y", "z", "90", "180", "270", "1", "0", "0" ]
        edits = self.trans + self.rot + self.rotC
        AngleValidator = QRegExpValidator(QRegExp("[-+]?[0-9]*[\.,]?[0-9]*"))
        for i in range(len(edits)):
            edits[i].setPlaceholderText(hints[i])
            edits[i].setValidator(AngleValidator)

    def makeButtons(self):
        ButtonsWidget = QWidget()
        buttonsLayout = QHBoxLayout()

        CancelBtn = QPushButton("Cancel")
        CancelBtn.clicked.connect(self.close)
        buttonsLayout.addWidget(CancelBtn)

        ApplyBtn = QPushButton("Apply")
        ApplyBtn.clicked.connect(self.apply)
        buttonsLayout.addWidget(ApplyBtn)

        ButtonsWidget.setLayout(buttonsLayout)
        self.layout.addWidget(ButtonsWidget,3,0)


    def close(self):
        self.setParent(None)

    def validation(self):
        if self.transformationType == "trans":
            vals = [self.trans[i].text() for i in range(3)]
            for val in vals:
                if val != "":
                    return True
        elif self.transformationType == "rot":
            vals = [self.rot[i].text() for i in range(3)]
            for val in vals:
                if val != "":
                    return True

    


    def parseVector(self, edits):
        vals = [edits[i].text()  for i in range(3)]
        # for i in vals:
        #     print("parsed: ", i)
        vals = list(map(lambda x: 0 if x=="" else x, vals))
        # for i in vals:
        #     print("parsed: ", i)
        vals = np.array(list(map(lambda x: float(x), vals)))
        # for i in vals:
        # print(vals)
        return vals


    def apply(self):
        if not self.validation():
            return


        if self.transformationType == "trans":
            translation = self.parseVector(self.trans)
            # print("!!!",self.element, self.transformationType, translation)
            self.applyTransformation(self.element, self.transformationType, translation)
        elif self.transformationType == "rot":
            rotation = self.parseVector(self.rot)
            rotationCenter = self.parseVector(self.rotC)
            self.applyTransformation(self.element, self.transformationType, rotation, rotationCenter)

            



if __name__ == '__main__':
    app = QApplication([])
    mw = TransformationWidget("Parabola_1",None)
    mw.setMinimumWidth(300)
    mw.show()
    app.exec_()