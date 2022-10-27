from PyQt5.QtWidgets import QWidget, QFormLayout, QLineEdit, QLabel, QPushButton, QComboBox, QHBoxLayout, QApplication
import sys
sys.path.append('../')
sys.path.append('../../')
import src.POPPy.System as system





class Form(QWidget):
    def __init__ (self):
        super().__init__()
        self.form = QFormLayout()

        self.fillForm()       
        self.setLayout(self.form)
        
    def fillForm(self):

        self.name = QLineEdit()
        self.name.setPlaceholderText("Parabola")
        self.form.addRow(QLabel("Name"),self.name)

        self.form.addRow(QLabel("Element Parameters"))

        self.pmode = QComboBox()
        self.pmode.addItems(["Manual", "Focus"])
        self.pmode.activated.connect(self.PModeChanged)
        self.form.addRow("Parameter mode", self.pmode)

        self.coefA,self.coefB = QLineEdit(),QLineEdit() 
        coefs = QHBoxLayout()
        coefs.addWidget(self.coefA)
        coefs.addWidget(self.coefB)
        self.form.addRow("Coefficients", coefs)

        self.vertex = [QLineEdit(),QLineEdit(),QLineEdit()]
        vertexLayout = QHBoxLayout()
        for e in self.vertex:
            vertexLayout.addWidget(e)
        self.form.addRow("Vertex", vertexLayout)

        self.focus = [QLineEdit(),QLineEdit(),QLineEdit()]
        focusLayout = QHBoxLayout()
        for e in self.focus:
            focusLayout.addWidget(e)
        self.form.addRow("Focus", focusLayout)
        self.PModeChanged()

        self.form.addRow(QLabel("Grid Parameters"))

        self.gmode = QComboBox()
        self.gmode.addItems(["Euclidean Parametrisation (x,y)", "Polar Parametrisation (u,v)"])
        self.gmode.activated.connect(self.GModeChanged)
        self.form.addRow("Grid mode", self.gmode)

        self.limX1,self.limX2 = QLineEdit(),QLineEdit()
        Xlims = QHBoxLayout()
        Xlims.addWidget(self.limX1)
        Xlims.addWidget(self.limX2)
        self.form.addRow("X-limits", Xlims)

        self.limY1,self.limY2 = QLineEdit(),QLineEdit() 
        Ylims = QHBoxLayout()
        Ylims.addWidget(self.limY1)
        Ylims.addWidget(self.limY2)
        self.form.addRow("Y-limits", Ylims)

        self.limU1,self.limU2 = QLineEdit(),QLineEdit() 
        Ulims = QHBoxLayout()
        Ulims.addWidget(self.limU1)
        Ulims.addWidget(self.limU2)
        self.form.addRow("U-limits", Ulims)

        self.limV1,self.limV2 = QLineEdit(),QLineEdit() 
        Vlims = QHBoxLayout()
        Vlims.addWidget(self.limV1)
        Vlims.addWidget(self.limV2)
        self.form.addRow("V-limits", Vlims)

        self.gridSizeX,self.gridSizeY = QLineEdit(),QLineEdit() 
        self.gridSizeX.setPlaceholderText("101")
        self.gridSizeY.setPlaceholderText("101")
        gridSizes = QHBoxLayout()
        gridSizes.addWidget(self.gridSizeX)
        gridSizes.addWidget(self.gridSizeY)
        self.form.addRow("Grid size", gridSizes)

        self.GModeChanged()

        self.addBtn = QPushButton("Add")
        self.addBtn.clicked.connect(self.GetDict)
        self.canselBtn = QPushButton("Cancel")
        self.canselBtn.clicked.connect(self.addNot)
        self.form.addRow(self.canselBtn, self.addBtn)

    def PModeChanged(self):
        if self.pmode.currentIndex() == 0:
            self.coefA.setEnabled(True)
            self.coefB.setEnabled(True)
            for i in range(3):
                self.vertex[i].setEnabled(False)
                self.focus[i].setEnabled(False)
        else:
            self.coefA.setEnabled(False)
            self.coefB.setEnabled(False)
            for i in range(3):
                self.vertex[i].setEnabled(True)
                self.focus[i].setEnabled(True)
            

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

    def GetDict(self):
        paramdict = {"name"     : None if self.name.text()=="" else self.name.text(),
                 "type"     : "Hyperbola",
                 "gridsize" : [int(self.gridSizeX.text()),int(self.gridSizeY.text())],
                #  "uvaxis"   : self.uvaxis,
                #  "units"    : self.units,
                #  "cRot"     : self.cRot.tolist(),
                #  "flip"     : self.flip,
                #  "sec"      : self.sec,
                #  "history"  : h_l
                 }
        if self.pmode.currentIndex() == 0:
            paramdict ["pmode"] = False
            paramdict ["coefs"] = [float(self.coefA.text()),float(self.coefB.text()),float(0)]
            if "vertex" in paramdict:
                del paramdict["vertex"]
            if "focus" in paramdict:
                del paramdict["focus"]
        else:
            paramdict ["pmode"] = True
            paramdict ["vertex"] = [float(self.vertex[0].text()),float(self.vertex[1].text()),float(self.vertex[2].text())]
            paramdict ["focus"] = [float(self.focus[0].text()),float(self.focus[1].text()),float(self.focus[2].text())]
            if "coefs" in paramdict:
                del paramdict["coefs"]

        if self.gmode.currentIndex() == 0:
            paramdict ["gmode"] = False
            paramdict["lims_x"] = [float(self.limX1.text()), float(self.limX2.text())]
            paramdict["lims_y"] = [float(self.limY1.text()), float(self.limY2.text())]
            if "lims_u" in paramdict:
                del paramdict["lims_u"]
            if "lims_v" in paramdict:
                del paramdict["lims_v"]
        else:
            paramdict ["gmode"] = True
            paramdict["lims_u"] = [float(self.limU1.text()), float(self.limU2.text())]
            paramdict["lims_v"] = [float(self.limV1.text()), float(self.limV2.text())]
            if "lims_x" in paramdict:
                del paramdict["lims_x"]
            if "lims_y" in paramdict:
                del paramdict["lims_y"]
        for k in paramdict:
            print(k, (15 - len(k))*" ",paramdict[k], (20 - len(str(paramdict[k])))*" ",type(paramdict[k]))


        
    def addNot(self):
        print("canceling")
        self.setParent(None)
        
    




if __name__ == '__main__':

    app = QApplication([])
    widget = Form()
    widget.setMaximumWidth(300)
    widget.show()
    app.exec_()