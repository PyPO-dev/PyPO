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
# win = SystemsColumn.SystemsWindow(["System_1", "sys2"])
Plane = [
    InputDescription(inType.string, "name" ),
    InputDescription(inType.floats, "coeffs", label="Coefficients", hints=[2], numFields=2),
    InputDescription(inType.integers, "gridsize", label="Grid Size", hints=[101,101], numFields=2),
    InputDescription(inType.dropdown, "pmode", label="Parameter Mode", subdict={
        "xy" : [InputDescription(inType.string, "xlims", oArray= True),
                InputDescription(inType.string, "ylims", oArray= True)],
        "uv" : [InputDescription(inType.string, "ulims", oArray= True),
                InputDescription(inType.string, "vlims", oArray= True)]
    })
]
win = FormGeneratorObjects(Plane, None)


with open('src/GUI/style.css') as f:
    style = f.read()
win.setStyleSheet(style)

win.show()
app.exec_()