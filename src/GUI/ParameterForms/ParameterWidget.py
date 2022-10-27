from PyQt5.QtWidgets import QWidget, QApplication, QComboBox, QVBoxLayout, QGridLayout, QLabel,QSpacerItem, QSizePolicy
# from PyQt5.QtCore import Qt

class ParameterWidget(QWidget):
    def __init__ (self):
        super().__init__()

        # parabolaDict = {"Coeficients":2, "X-Limits": 2, }







if __name__ == '__main__':

    app = QApplication([])
    widget = ParameterWidget()
    # widget.setMaximumWidth(300)
    widget.show()
    app.exec_()