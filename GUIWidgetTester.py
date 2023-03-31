import sys
from PyQt5.QtWidgets import QApplication
from src.GUI import *
from src.GUI.Acccordion import Accordion
from src.GUI.ElementWidget import ElementWidget
from GUI.ParameterForms.simpleInputWidgets import ElementSelectionWidget
import traceback

from src.GUI.WorkSpace import Workspace
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
    # InputDescription(inType.static, label = "static no output", staticValue="code Filled Value", toolTip="tt: static no output"),
    # InputDescription(inType.static, "static do output", staticValue="codeFilledValue", toolTip="tt: "),
    # InputDescription(inType.static, "hidden field", staticValue="hidden value", hidden=True, toolTip="tt: "),
    # InputDescription(inType.checkbox, "checkBox", staticValue="pla", toolTip="tt: "),
    # InputDescription(inType.vectorStrings, "string", hints=["Enter name"], toolTip="tt: Enter name of reflector", prefill=True),
    # InputDescription(inType.vectorIntegers, "integers", hints=[2], toolTip="tt: vector of integers"),
    # InputDescription(inType.static, "static do output", staticValue="codeFilledValue", toolTip="tt: static do output"),
    # InputDescription(inType.static, "hidden field", staticValue="hidden value", hidden=True),
    # InputDescription(inType.checkbox, "checkBox", staticValue="pla", prefill=True),
    # InputDescription(inType.vectorStrings, "string", hints=["a", "&&&"], prefill=True, numFields= 2),
    # InputDescription(inType.vectorIntegers, "integers", hints=[599756], prefill=True),
    # InputDescription(inType.vectorFloats, "floats", label="Coefficients", hints=[1.,2.,3e15], numFields=3, oArray=True, prefill=True, toolTip="this one is already prefilled"),
    # InputDescription(inType.radio, "radio buttons", options= ["a", "b", "c"], hints= ["A", "B", "C"]),
    # InputDescription(inType.dropdown, "radio buttons", options= ["a", "b", "c"], hints= ["A", "B", "C"]),
    # InputDescription(inType.xyzradio, "Axes selector"),
    # InputDescription(inType.elementSelector, "outname", options= ["parabola1", "frame1", "other object", "frame1", "other object", "frame1", "other object", "frame1", "other object"],toolTip="Select elements to apply transformations on."),
    InputDescription(inType.dynamicDropdown, "Axes selector", label = "dynamic dropdown", toolTip="tt: dropdown", subdict={
        "a" : [
            InputDescription(inType.vectorStrings, "string", hints=["Enter name"], toolTip="tt: 0"),
            InputDescription(inType.vectorIntegers, "integers", hints=[2], toolTip="tt: 1")
        ],
        "b" : [
            InputDescription(inType.static, label = "static no output", staticValue="code Filled Value", toolTip="tt: 2"),
            InputDescription(inType.vectorStrings, "string", hints=["Enter name"], toolTip="tt: 3"),
            InputDescription(inType.vectorFloats, "floats", label="Coefficients", hints=[1,2,3], numFields=3, oArray=True, toolTip="tt: 3")
        ],
    })
    # InputDescription(inType.dynamicRadio, "Axes selector2", label = "dynamic dropdown", toolTip="tt: radio", subdict={
    #     "a" : [
    #         InputDescription(inType.vectorStrings, "string2", hints=["Enter name"], toolTip="tt: 5"),
    #         InputDescription(inType.vectorIntegers, "integers2", hints=[2], toolTip="tt: 6")
    #     ],
    #     "b" : [
    #         InputDescription(inType.static, label = "static no output2", staticValue="code Filled Value", toolTip="tt: 7"),
    #         InputDescription(inType.vectorStrings, "string2", hints=["Enter name"], toolTip="tt: 8"),
    #         InputDescription(inType.vectorFloats, "floats2", label="Coefficients", hints=[1,2,3], numFields=3, toolTip="tt: 9", oArray=True)
    #     ]
    # })
]


def reader():
    try:
        print(win.read())
    except Exception as err:
        print(err)
        traceback.print_tb(err.__traceback__)


win = FormGenerator(Plane, reader,okText="OK")

# +   [InputDescription(inType.vectorStrings, "name", label="Name of frame", numFields=1),
#             InputDescription(inType.vectorIntegers, "nRays", label="# of rays", hints=[0], numFields=1),
#             InputDescription(inType.vectorFloats, "n", label="Refractive index of medium", hints=[1], numFields=1),
#             InputDescription(inType.vectorFloats, "lam", label="Wavelength", hints=[1], numFields=1),
#             InputDescription(inType.vectorFloats, "x0", label="X beamwaist", hints=[5], numFields=1),
#             InputDescription(inType.vectorFloats, "y0", label="Y beamwaist", hints=[5], numFields=1),
#             InputDescription(inType.vectorFloats, "tChief", label="Chief ray tilt", hints=[0,0,1], numFields=3, oArray=True),
#             InputDescription(inType.vectorFloats, "oChief", label="Chief ray origin", hints=[0,0,0], numFields=3, oArray=True),
#             InputDescription(inType.dynamicRadio, "setseed", label="Set seed", subdict={
#                 "random" : [],
#                 "set" : [InputDescription(inType.vectorIntegers, "seed", label="", hints=[0], numFields=1)]
#             })] 


# inp = 
# print(type(inp), inp)
# win = ElementSelectionWidget(inp, reader)

# win = Accordion()
# win.reflectors.addWidget(ElementWidget("Refl1", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl2", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl3", lambda x:0, lambda x:0))
# win.reflectors.addWidget(ElementWidget("Refl4", lambda x:0, lambda x:0))
# win.RayTraceFrames.addWidget(ElementWidget("Frame2", lambda x:0,lambda x:0,lambda x:0))

# win = Workspace()
# win.elementSpace.Reflectors.addWidget(ElementWidget("Frame1", lambda x:0,lambda x:0,lambda x:0))

with open('src/GUI/style.css') as f:
    style = f.read()
win.setStyleSheet(style)

win.show()
app.exec_()
