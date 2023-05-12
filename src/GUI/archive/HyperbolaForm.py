from PySide2.QtWidgets import QWidget, QFormLayout, QLineEdit, QLabel, QPushButton, QComboBox, QHBoxLayout, QApplication
import sys
import numpy as np

sys.path.append('../')
sys.path.append('../../')

class Form(QWidget):
    def __init__ (self, addElement):
        super().__init__()

        self.addElementAction = addElement
        self.form = QFormLayout()

        self.fillForm()       
        self.setLayout(self.form)
        
    def fillForm(self):

        self.name = QLineEdit()
        self.name.setPlaceholderText("Hyperbola")
        self.form.addRow(QLabel("Name"),self.name)

        self.form.addRow(QLabel("Element Parameters"))

        self.pmode = QComboBox()
        self.pmode.addItems(["Manual", "focus_1"])
        self.pmode.activated.connect(self.PModeChanged)
        self.form.addRow("Parameter mode", self.pmode)

        self.coefA,self.coefB,self.coefC = QLineEdit(),QLineEdit(),QLineEdit() 
        coeffs = QHBoxLayout()
        coeffs.addWidget(self.coefA)
        coeffs.addWidget(self.coefB)
        coeffs.addWidget(self.coefC)
        self.form.addRow("Coefficients", coeffs)

        self.vertex = QLineEdit()
        self.form.addRow("Vertex", self.vertex)

        self.focus_1 = QLineEdit()
        self.form.addRow("focus_1", self.focus_1)
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
            self.vertex.setEnabled(False)
            self.focus_1.setEnabled(False)
        else:
            self.coefA.setEnabled(False)
            self.coefB.setEnabled(False)
            self.vertex.setEnabled(True)
            self.focus_1.setEnabled(True)
            

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
            
        else:
            self.limX1.setEnabled(False)
            self.limX2.setEnabled(False)
            self.limY1.setEnabled(False)
            self.limY2.setEnabled(False)
            self.limU1.setEnabled(True)
            self.limU2.setEnabled(True)
            self.limV1.setEnabled(True)
            self.limV2.setEnabled(True)
            # print("dis2")

    def GetDict(self):
        paramdict = {"name"     : None if self.name.text()=="" else self.name.text(),
                 "type"     : "Hyperbola",
                 "gridsize" : [int(self.gridSizeX.text()),int(self.gridSizeY.text())],
                #  "uvaxis"   : self.uvaxis,
                #  "units"    : self.units,
                #  "cRot"     : self.cRot.tolist(),
                 "flip"     : False,
                #  "sec"      : self.sec,
                #  "history"  : h_l
                 }
        if self.pmode.currentIndex() == 0:
            paramdict ["pmode"] = "manual"
            paramdict ["coeffs"] = np.array([float(self.coefA.text()),float(self.coefB.text()),float(self.coefC.text())])
            if "vertex" in paramdict:
                del paramdict["vertex"]
            if "focus_1" in paramdict:
                del paramdict["focus_1"]
        else:
            paramdict ["pmode"] = "focus"
            paramdict ["vertex"] = np.array([float(self.vertex[0].text()),float(self.vertex[1].text()),float(self.vertex[2].text())])
            paramdict ["focus_1"] = np.array([float(self.focus_1[0].text()),float(self.focus_1[1].text()),float(self.focus_1[2].text())])
            if "coeffs" in paramdict:
                del paramdict["coeffs"]

        if self.gmode.currentIndex() == 0:
            paramdict ["gmode"] = "xy"
            paramdict["lims_x"] = [float(self.limX1.text()), float(self.limX2.text())]
            paramdict["lims_y"] = [float(self.limY1.text()), float(self.limY2.text())]
            if "lims_u" in paramdict:
                del paramdict["lims_u"]
            if "lims_v" in paramdict:
                del paramdict["lims_v"]
        else:
            paramdict ["gmode"] = "uv"
            paramdict["lims_u"] = [float(self.limU1.text()), float(self.limU2.text())]
            paramdict["lims_v"] = [float(self.limV1.text()), float(self.limV2.text())]
            if "lims_x" in paramdict:
                del paramdict["lims_x"]
            if "lims_y" in paramdict:
                del paramdict["lims_y"]


        # for k in paramdict:
        #     print(k, (15 - len(k))*" ",paramdict[k], (20 - len(str(paramdict[k])))*" ",type(paramdict[k]))

        self.addElementAction(paramdict)

        
    def addNot(self):
        print("canceling")
        self.setParent(None)
        
    




if __name__ == '__main__':

    app = QApplication([])
    widget = Form()
    widget.setMaximumWidth(300)
    widget.show()
    app.exec_()