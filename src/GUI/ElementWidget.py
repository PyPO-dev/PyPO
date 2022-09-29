from PyQt5.QtWidgets import QLabel, QApplication, QWidget, QHBoxLayout, QMainWindow, \
                                    QPushButton, QSpacerItem, QSizePolicy, QDialog, \
                                    QAction
                                        
from PyQt5.QtGui import QFont, QIcon, QCursor 
from PyQt5.QtCore import QMargins, Qt
from ElementOptionsLayout import ElementOptionsLayout
import sys
sys.path.append('../')
sys.path.append('../../')
import time



class ElementWidget(QWidget):
    def __init__ (self, p, element):
        super().__init__(p)
        layout = QHBoxLayout()
        label = QLabel(element)
        self.optionsOpen = False

        # label.setFont(QFont('Times New Roman', 18))
        label.setStyleSheet("border-top-left-radius :15px;"
                            " border-top-right-radius : 0px; "
                            "border-bottom-left-radius : 15px; "
                            "border-bottom-right-radius : 0px")
        label.setFixedSize(200,39)

        self.btn = QPushButton("btn")
        self.btn.clicked.connect(self._openOptionsMenu)
        self.btn.setIcon(QIcon("src/GUI/Images/dots.png"))
        self.btn.setFixedSize(50,39)
        
        self.btn.setStyleSheet("border-top-left-radius :0px;"
                            " border-top-right-radius : 15px; "
                            "border-bottom-left-radius : 0px; "
                            "border-bottom-right-radius : 15px")

        layout.setSpacing(0)
        layout.addWidget(label)
        layout.addWidget(self.btn)

        self.setLayout(layout)
        self.setFixedSize(250,60)
  
        self.setStyleSheet("*{background:#6715a5;}"
                            "*:hover{background:#a51554; }")

            

        ### debug style
        # self.setStyleSheet("border: 2px dotted #ff6550;\
        #                     background: #ffffff;")     

    def _openOptionsMenu(self):
        if self.optionsOpen:
            self.closeOptionsMenu()
            return
        
        self.optionsOpen = True
        self.dlg = QDialog(self)
        self.dlg.setX = QCursor.pos().x
        self.dlg.y = QCursor.pos().y
        self.dlg.setLayout(ElementOptionsLayout(self.closeOptionsMenu))
        self.dlg.setWindowFlag(Qt.FramelessWindowHint)
        
        self.dlg.setGeometry(QCursor.pos().x(), QCursor.pos().y() ,50,50)
        self.dlg.show()
        

    def closeOptionsMenu(self):
        self.optionsOpen = False
        self.dlg.close()
        


if __name__ == "__main__":
    app = QApplication([])
    window = QMainWindow()
    window.resize(500,800)
    wid = ElementWidget(window,"Parabola_0")
    
    window.show()
    app.exec_()