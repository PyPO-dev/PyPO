import sys
from PyQt5.QtWidgets import QApplication
from src.GUI import *
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ElementWidget
# from src.GUI.SystemsColumn import *
# from src.GUI.selfClosingDialog import *
# from src.GUI.selfClosingDialog_HoverableBtn import *
# from src.GUI.PlotScreen import *
# from src.GUI.parabolaformInWidgets import *
# from src.GUI.MPLCanvas import *
# from src.GUI.MainWindow import *
# from src.GUI.MainWidget import *
# from src.GUI.ElementWidget_Systems import *
# from src.GUI.ElementsColumn implambda x :print(x)t *

app = QApplication(sys.argv)

# win = ElementWidget_Systems.SystemWidget("System_1")
# win = SystemsColumn.SystemsWindow(["System_1", "sys2"])
Plane = [
    # InputDescription(inType.static, label = "static no output", staticValue="code Filled Value"),
    # InputDescription(inType.static, "static do output", staticValue="codeFilledValue"),
    # InputDescription(inType.static, "hidden field", staticValue="hidden value", hidden=True),
    # InputDescription(inType.checkbox, "checkBox", staticValue="pla"),
    # InputDescription(inType.vectorStrings, "string", hints=["Enter name"]),
    # InputDescription(inType.vectorIntegers, "integers", hints=[2]),
    # InputDescription(inType.vectorFloats, "floats", label="Coefficients", hints=[1,2,3], numFields=3, oArray=True),
    # InputDescription(inType.radio, "radio buttons", options= ["a", "b", "c"], hints= ["A", "B", "C"]),
    # InputDescription(inType.dropdown, "radio buttons", options= ["a", "b", "c"], hints= ["A", "B", "C"]),
    # InputDescription(inType.xyzradio, "Axes selector"),
    InputDescription(inType.dynamicDropdown, "Axes selector", label = "dynamic dropdown", subdict={
        "a" : [
            InputDescription(inType.vectorStrings, "string", hints=["Enter name"]),
            InputDescription(inType.vectorIntegers, "integers", hints=[2])
        ],
        "b" : [
            InputDescription(inType.static, label = "static no output", staticValue="code Filled Value"),
            InputDescription(inType.vectorStrings, "string", hints=["Enter name"]),
            InputDescription(inType.vectorFloats, "floats", label="Coefficients", hints=[1,2,3], numFields=3, oArray=True)
        ],
    }),
    InputDescription(inType.dynamicRadio, "Axes selector2", label = "dynamic dropdown", subdict={
        "a" : [
            InputDescription(inType.vectorStrings, "string2", hints=["Enter name"]),
            InputDescription(inType.vectorIntegers, "integers2", hints=[2])
        ],
        "b" : [
            InputDescription(inType.static, label = "static no output2", staticValue="code Filled Value"),
            InputDescription(inType.vectorStrings, "string2", hints=["Enter name"]),
            InputDescription(inType.vectorFloats, "floats2", label="Coefficients", hints=[1,2,3], numFields=3, oArray=True)
        ],
    })
]
def reader():
    print(win.read())

win = FormGenerator(Plane, reader)

# win = Accordion()
# win.reflectors.addWidget(ElementWidget("Refl1", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl2", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl3", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl4", lambda x:0, lambda x:0))
# win.RayTraceFrames.addWidget(ElementWidget("Frame1", lambda x:0,lambda x:0,lambda x:0))
# win.RayTraceFrames.addWidget(ElementWidget("Frame2", lambda x:0,lambda x:0,lambda x:0))

with open('src/GUI/style.css') as f:
    style = f.read()
win.setStyleSheet(style)

win.show()
app.exec_()
