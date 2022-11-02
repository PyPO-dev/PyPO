import sys
from PyQt5.QtWidgets import QApplication
from src.GUI import *

# from src.GUI.SystemsColumn import *
# from src.GUI.selfClosingDialog import *
# from src.GUI.selfClosingDialog_HoverableBtn import *
# from src.GUI.PlotScreen import *
# from src.GUI.parabolaformInWidgets import *
# from src.GUI.MPLCanvas import *
# from src.GUI.MainWindow import *
# from src.GUI.MainWidget import *
# from src.GUI.ElementWidget_Systems import *
# from src.GUI.ElementsColumn import *

app = QApplication(sys.argv)

# win = ElementWidget_Systems.SystemWidget("System_1")
win = SystemsColumn.SystemsWindow(["System_1", "sys2"])


with open('src/GUI/style.css') as f:
    style = f.read()
win.setStyleSheet(style)

win.show()
app.exec_()