from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMenuBar, QMenu, QGridLayout, QWidget, QSizePolicy, QPushButton, QVBoxLayout, QHBoxLayout, QAction, QTabWidget, QTabBar, QScrollArea
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject


class Waiter(QObject):
    finished = pyqtSignal()

    def setProcess(self, process):
        self.process = process

    def run(self):
        self.process.join()
        print("waiter: Process joined")
        self.finished.emit()
