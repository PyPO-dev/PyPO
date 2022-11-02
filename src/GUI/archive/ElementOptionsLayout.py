from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSpacerItem, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout

class ElementOptionsLayout(QVBoxLayout):
    def __init__ (self, cls):
        super().__init__()
        self.setSpacing(0)
        self.setContentsMargins(0,0,0,0)

        self.close = cls

        btn1 = QPushButton("Transform")
        btn2 = QPushButton("Edit")
        btn3 = QPushButton("Plot")

        btn1.clicked.connect(cls)
        btn2.clicked.connect(cls)
        btn3.clicked.connect(cls)

        self.addWidget(btn1)
        self.addWidget(btn2)
        self.addWidget(btn3)

if __name__ == "__main__":
    app = QApplication([])
    window = QWidget()
    
    window.setLayout(ElementOptionsLayout())
    
    window.show()
    app.exec_()