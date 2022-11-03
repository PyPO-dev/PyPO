from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QFormLayout, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt

class FormWidget(QWidget):
    def __init__ (self):
        super().__init__()

        ### Setup Ui ###
        self.form = QFormLayout()
        self.form.addRow(QLabel("Element Type"), self.ElementSelector())
        
        # self.form.setAlignment(Qt.AlignTop)
        self.setLayout(self.form)

    def ElementSelector(self):
        self.ElementSelector = QComboBox()
        self.ElementSelector.addItems([
            "--select Element type--"
            , "Reflector"
            , "Camera"
            , "RayTracer"
            ])
        self.ElementSelector.currentIndexChanged.connect(self.ElementSelected)
        return self.ElementSelector

    def ElementSelected(self):
        if self.ElementSelector.currentIndex() == 1:
            if hasattr(self, "ReflectorSelector"):
                self.ReflectorSelector.setParent(None)
            self.form.addWidget(self._makeReflectorSelector(),1,0,1,2)

    def _makeReflectorSelector(self):

        self.ReflectorSelector = QComboBox()
        self.ReflectorSelector.addItems([
            "--select Reflector type--"
            , "Parabola"
            , "Hyperbola"
            , "Ellipse"
            ])
        self.ReflectorSelector.currentIndexChanged.connect(self.ReflectorSelected)
        return self.ReflectorSelector

    def ReflectorSelected(self):
        if self.ReflectorSelector.currentIndex() == 1:
            self.form.addWidget(QLabel("ParabolaForm"),2,0,1,2)
        if self.ReflectorSelector.currentIndex() == 2:
            self.form.addWidget(QLabel("HeyperbolaForm"),2,0,1,2)
        if self.ReflectorSelector.currentIndex() == 3:
            self.form.addWidget(QLabel("EllipseForm"),2,0,1,2)

    
        








if __name__ == '__main__':

    app = QApplication([])
    widget = FormWidget()
    widget.setMaximumWidth(300)
    widget.show()
    app.exec_()