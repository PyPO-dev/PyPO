from PyQt5.QtWidgets import QWidget, QScrollArea, QApplication, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt
import sys

class Accordion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)
        layout.setAlignment(Qt.AlignTop)

        self.reflectors = AccordionSection("Reflectors")
        self.RayTraceFrames = AccordionSection("Ray Trace Frames")
        self.POFields = AccordionSection("PO Fields")
        self.POCurrents = AccordionSection("PO Currents")
        layout.addWidget(self.reflectors)
        layout.addWidget(self.RayTraceFrames)
        layout.addWidget(self.POFields)
        layout.addWidget(self.POCurrents)
        
class AccordionSection(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)

        ### MainLayout ###
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        ### Header ###
        Header = QWidget()
        headerLayout = QHBoxLayout()
        Header.setLayout(headerLayout)

        textLabel = QLabel(text)
        textLabel.setAlignment(Qt.AlignLeft)
        self.arrowLabel = QLabel()
        self.arrowLabel.setAlignment(Qt.AlignRight)
        headerLayout.addWidget(textLabel)
        headerLayout.addWidget(self.arrowLabel)

        Header.mouseReleaseEvent = self.toggle
        layout.addWidget(Header)

        ### Content ###
        self.content = QWidget()
        self.contentLayout = QVBoxLayout()
        # self.contentLayout.setContentsMargins(0,0,0,0)
        self.content.setLayout(self.contentLayout)
        layout.addWidget(self.content)

        # self.hideContent()
        self.showContent()

    def toggle(self, event):
        if self.content.isVisible():
            self.hideContent()
        else: 
            self.showContent()

    def hideContent(self):
        self.content.hide()
        self.arrowLabel.setText("▽")

    def showContent(self):
        self.content.show()
        self.arrowLabel.setText("△")
    
    def addWidget(self, widget):
        self.contentLayout.addWidget(widget)


