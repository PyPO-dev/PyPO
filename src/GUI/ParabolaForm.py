from PyQt5 import QtWidgets as qtw
import sys
sys.path.append('../')
sys.path.append('../../')
import POPPy.System as system





class ParabolaFormLayout(qtw.QFormLayout):
    def __init__ (self, System):
        super().__init__()

        self.System = System

        self.name = qtw.QLineEdit()
        self.name.setPlaceholderText("Parabola")
        self.addRow(qtw.QLabel("Name"),self.name)

        self.addRow(qtw.QLabel("Element Parameters"))

        self.pmode = qtw.QComboBox()
        self.pmode.addItems(["Manual", "Focus"])
        self.pmode.activated.connect(self.PModeChanged)
        self.addRow("Parameter mode", self.pmode)

        self.coefA,self.coefB = qtw.QLineEdit(),qtw.QLineEdit() 
        self.coefs = qtw.QHBoxLayout()
        self.coefs.addWidget(self.coefA)
        self.coefs.addWidget(self.coefB)
        self.addRow("Coefficients", self.coefs)

        self.vertex = qtw.QLineEdit()
        self.addRow("Vertex", self.vertex)

        self.focus = qtw.QLineEdit()
        self.addRow("Focus", self.focus)
        self.PModeChanged()


        self.addRow(qtw.QLabel("Grid Parameters"))

        self.gmode = qtw.QComboBox()
        self.gmode.addItems(["Euclidean Parametrisation", "Polar Parametrisation"])
        self.gmode.activated.connect(self.GModeChanged)
        self.addRow("Parameter mode", self.gmode)

        self.limX1,self.limX2 = qtw.QLineEdit(),qtw.QLineEdit() 
        self.Xlims = qtw.QHBoxLayout()
        self.Xlims.addWidget(self.limX1)
        self.Xlims.addWidget(self.limX2)
        self.addRow("X-limits", self.Xlims)

        self.limY1,self.limY2 = qtw.QLineEdit(),qtw.QLineEdit() 
        self.Ylims = qtw.QHBoxLayout()
        self.Ylims.addWidget(self.limY1)
        self.Ylims.addWidget(self.limY2)
        self.addRow("Y-limits", self.Ylims)

        self.limU1,self.limU2 = qtw.QLineEdit(),qtw.QLineEdit() 
        self.Ulims = qtw.QHBoxLayout()
        self.Ulims.addWidget(self.limU1)
        self.Ulims.addWidget(self.limU2)
        self.addRow("U-limits", self.Ulims)

        self.limV1,self.limV2 = qtw.QLineEdit(),qtw.QLineEdit() 
        self.Vlims = qtw.QHBoxLayout()
        self.Vlims.addWidget(self.limV1)
        self.Vlims.addWidget(self.limV2)
        self.addRow("V-limits", self.Vlims)

        self.gridSizeX,self.gridSizeY = qtw.QLineEdit(),qtw.QLineEdit() 
        self.gridSizeX.setPlaceholderText("101")
        self.gridSizeY.setPlaceholderText("101")
        self.gridSizes = qtw.QHBoxLayout()
        self.gridSizes.addWidget(self.gridSizeX)
        self.gridSizes.addWidget(self.gridSizeY)
        self.addRow("Grid size", self.gridSizes)

        self.GModeChanged()

        self.addBtn = qtw.QPushButton("Add")
        self.addBtn.clicked.connect(self.addElement)
        self.canselBtn = qtw.QPushButton("Cancel")
        self.canselBtn.clicked.connect(self.addNot)
        self.addRow(self.canselBtn, self.addBtn)

        
        # self.setStyleSheet("background: #D9D9D9;")

    def PModeChanged(self):
        if self.pmode.currentIndex() == 0:
            self.coefA.setEnabled(True)
            self.coefB.setEnabled(True)
            self.vertex.setEnabled(False)
            self.focus.setEnabled(False)
        else:
            self.coefA.setEnabled(False)
            self.coefB.setEnabled(False)
            self.vertex.setEnabled(True)
            self.focus.setEnabled(True)
            

    def GModeChanged(self):
        if self.gmode.currentIndex() == 0:
            self.limX1.setEnabled(True)
            self.limX2.setEnabled(True)
            self.limY1.setEnabled(True)
            self.limY2.setEnabled(True)
            self.limU1.setEnabled(False)
            self.limU2.setEnabled(False)
            self.limV1.setEnabled(False)
            self.limV2.setEnabled(False)
            print("dis1")
            
        else:
            self.limX1.setEnabled(False)
            self.limX2.setEnabled(False)
            self.limY1.setEnabled(False)
            self.limY2.setEnabled(False)
            self.limU1.setEnabled(True)
            self.limU2.setEnabled(True)
            self.limV1.setEnabled(True)
            self.limV2.setEnabled(True)
            print("dis2")

    def addElement(self):
        
        self.System.addParabola([float(self.coefA.text()), float(self.coefB.text())],
         [float(self.limX1.text()),float(self.limX2.text())], 
         [float(self.limY1.text()),float(self.limY2.text())], 
         [int(self.gridSizeX.text()), int(self.gridSizeY.text())])
        

    def addNot(self):
        print("canceling")
        for i in reversed(range(self.count())): 
            print(self.itemAt(i)) #.widget().setParent(None)
        # print(self.getLayoutPosition())
        




if __name__ == '__main__':

    app = qtw.QApplication([])
    widget = qtw.QWidget()
    widget.setLayout(ParabolaFormLayout(None))
    widget.setMaximumWidth(300)
    widget.show()
    app.exec_()