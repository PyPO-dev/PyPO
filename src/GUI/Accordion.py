from PySide2.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide2.QtCore import Qt

##
# @file An implementation of an accordion component.
#  

##
# Class to implement an Accordion widget.
class Accordion(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setAlignment(Qt.AlignTop)

        self.POFields = self.AccordionSection("Fields")
        self.POCurrents = self.AccordionSection("Currents")
        self.SPOFields = self.AccordionSection("Scalar Fields")
        layout.addWidget(self.POFields)
        layout.addWidget(self.POCurrents)
        layout.addWidget(self.SPOFields)
            
    ##
    # Define section of Accordian.
    class AccordionSection(QWidget):
        def __init__(self, text, parent=None):
            super().__init__(parent)

            ### MainLayout ###
            layout = QVBoxLayout()
            layout.setAlignment(Qt.AlignTop)
            self.setLayout(layout)

            ### Header ###
            Header = QWidget()
            headerLayout = QHBoxLayout(Header)

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
            self.contentLayout = QVBoxLayout(self.content)
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


