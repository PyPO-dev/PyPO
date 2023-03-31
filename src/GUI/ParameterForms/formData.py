from src.GUI.ParameterForms.InputDescription import inType, InputDescription

def xy_opts():
    return [InputDescription(inType.vectorFloats, "lims_x", label="X limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "lims_y", label="Y limits", oArray=True, numFields=2),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors")]

def uv_opts():
    return [InputDescription(inType.vectorFloats, "lims_u", label="U limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "lims_v", label="V limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "gcenter", label="XY center", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "ecc_uv", label="UV eccentricity", numFields=1),
            InputDescription(inType.vectorFloats, "rot_uv", label="UV position angle", numFields=1),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors")]

def AoE_opts():
    return [InputDescription(inType.vectorFloats, "lims_Az", label="Azimuth limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "lims_El", label="Elevation limits", oArray=True, numFields=2)]

def focus_opts_hyp_ell():
    return [
        InputDescription(inType.vectorFloats, "focus_1", label="Upper focus xyz", oArray=True, numFields=3),
            InputDescription(inType.vectorFloats, "focus_2", label="Lower focus xyz", oArray=True, numFields=3),
            InputDescription(inType.vectorFloats, "ecc", label="Eccentricity", numFields=1)
    ]

def makeParabolaInp():
    return [
        InputDescription(inType.vectorStrings, "name"),
        InputDescription(inType.dynamicRadio, "pmode", label="Parameter mode", subdict={
            "focus"     : [InputDescription(inType.vectorFloats, "focus_1", label="Focus xyz", oArray=True, numFields=3),
                            InputDescription(inType.vectorFloats, "vertex", label="Vertex xyz", oArray=True, numFields=3)],
            "manual"    : [InputDescription(inType.vectorFloats, "coeffs", label="AB coefficients", oArray=True, numFields=2)]
            }),
        InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
        InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
        InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subdict={
            "xy" : xy_opts(),
            "uv" : uv_opts()
        })
    ]

def makeHyperbolaEllipseInp():
    return [InputDescription(inType.vectorStrings, "name"),
            InputDescription(inType.dynamicRadio, "pmode", label="Parameter mode", subdict={
                "focus"     : focus_opts_hyp_ell(),
                "manual"    : [InputDescription(inType.vectorFloats, "coeffs", label="ABC coefficients", oArray=True, numFields=3)]
                }),
            InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
            InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subdict={
                "xy" : xy_opts(),
                "uv" : uv_opts()
            })]

def makeQuadricSurfaceInp():
    return [
        InputDescription(inType.dynamicDropdown, "type", subdict={
            "Parabola": makeParabolaInp(),
            "Hyperbola": makeHyperbolaEllipseInp(),
            "Ellipse": makeHyperbolaEllipseInp()
        })
    ]

def makePlaneInp():
    return [InputDescription(inType.vectorStrings, "name"),
            InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subdict={
                "xy" : xy_opts(),
                "uv" : uv_opts(),
                "AoE" : AoE_opts()
            })]
            #InputDescription(inType.integers, "flip", label="Flip normal vectors", numFields=1)]

def makeTransformationForm(elementName, obj="element"):
    return[
        InputDescription(inType.static, obj, staticValue=elementName),
        InputDescription(inType.radio, "mode", label="Transformation mode", options=[
            "Relative", "Absolute"
            ]),
        InputDescription(inType.dynamicRadio, "type", subdict={
            "Translation":[
                InputDescription(inType.vectorFloats, "vector", label="Translation Vector", hints=["x","y","z"], numFields=3,oArray=True)],
            "Rotation": [
                InputDescription(inType.vectorFloats, "vector", label="Rotation Vector", hints=["x","y","z"], numFields=3,oArray=True),
                InputDescription(inType.vectorFloats, "pivot", label="Center of Rotation", hints=["x","y","z"], numFields=3,oArray=True)
                ]
        })
    ]

def initTubeFrameInp():
    return [InputDescription(inType.vectorStrings, "name", label="Name of frame", numFields=1),
            InputDescription(inType.vectorIntegers, "nRays", label="# of rays", hints=[0], numFields=1),
            InputDescription(inType.vectorIntegers, "nRing", label="# of rings", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "angx0", label="X-apex angle", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "angy0", label="Y-apex angle", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "x0", label="X radius of outer ring", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "y0", label="Y radius of outer ring", hints=[0], numFields=1)
            ]

def initGaussianFrameInp():

    return [InputDescription(inType.vectorStrings, "name", label="Name of frame", numFields=1),
            InputDescription(inType.vectorIntegers, "nRays", label="# of rays", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "n", label="Refractive index of medium", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "x0", label="X beamwaist", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "y0", label="Y beamwaist", hints=[5], numFields=1),
            InputDescription(inType.dynamicRadio, "setseed", label="Set seed", subdict={
                "random" : [],
                "set" : [InputDescription(inType.vectorIntegers, "seed", label="", hints=[0], numFields=1)]
            })]

def plotFrameInp(frameDict):
    sublist_frames = []
    if frameDict:
        for key in frameDict.keys():
            sublist_frames.append(key)
    
    plotFrame = [
            InputDescription(inType.dropdown, "frame", label="Frame", options = sublist_frames),
            InputDescription(inType.xyzradio, "project", label="Abscissa - ordinate")
            ]

    return plotFrame

def plotFrameOpt(frameName):
    plotFrame = [
            InputDescription(inType.static, "frame", label="Frame", staticValue=frameName),
            InputDescription(inType.xyzradio, "project", label="Abscissa - ordinate")
            ]

    return plotFrame

def propRaysInp(frameDict, elemDict):
    sublist_frames = []
    sublist_target = []
    if frameDict:
        for key, item in frameDict.items():
            sublist_frames.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            sublist_target.append(key)
    
    sublist_dev = ["CPU", "GPU"]

    propRays = [
            InputDescription(inType.dropdown, "fr_in", label="Input frame", options = sublist_frames),
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.vectorStrings, "fr_out", label="Name of output frame", numFields=1),
            InputDescription(inType.vectorFloats, "tol", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "t0", label="Initial guess", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propRays
    
def initPSInp(elemDict):
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initPS = [
            InputDescription(inType.dropdown, "surface", label="Point source surface", options = sublist_surf),
            InputDescription(inType.vectorStrings, "name", label="Beam name", numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength of radiation", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "phase", label="Phase", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "pol", label="Polarization", hints=[1,0,0], numFields=3, oArray=True)
            ]
    
    return initPS

#TODO: this file should not contain functional code It should only provide data
def initSPSInp(elemDict):
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initSPS = [
            InputDescription(inType.dropdown, "surface", label="Point source surface", options = sublist_surf),
            InputDescription(inType.vectorStrings, "name", label="Beam name", numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength of radiation", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "phase", label="Phase", hints=[0], numFields=1),
            ]
    
    return initSPS

def initGaussianInp(elemDict):
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initGauss = [
            InputDescription(inType.dropdown, "surface", label="Gaussian beam surface", options = sublist_surf),
            InputDescription(inType.vectorStrings, "name", label="Beam name", numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength of radiation", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "w0x", label="Beamwaist in X", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "w0y", label="Beamwaist in Y", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "n", label="Refractive index", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "z", label="Focal distance", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "pol", label="Polarization", hints=[1,0,0], numFields=3, oArray=True)
            ]
    
    return initGauss

def initSGaussianInp(elemDict):
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initSGauss = [
            InputDescription(inType.dropdown, "surface", label="Gaussian beam surface", options = sublist_surf),
            InputDescription(inType.vectorStrings, "name", label="Beam name", numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength of radiation", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "w0x", label="Beamwaist in X", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "w0y", label="Beamwaist in Y", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "n", label="Refractive index", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "z", label="Focal distance", hints=[0], numFields=1),
            ]
    
    return initSGauss

def plotField(fieldName):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    plotField = [
            InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
            InputDescription(inType.dropdown, "comp", label="Component", options = complist),
            InputDescription(inType.xyzradio, "project", label="Abscissa - ordinate")
            ]

    return plotField

def plotSField(fieldName, gmode):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    if gmode != 2:
        plotSField = [
               InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                InputDescription(inType.xyzradio, "project", label="Abscissa - ordinate")
                ]

    else:
        plotSField = [
                InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                InputDescription(inType.static, "project", staticValue="xy", hidden=True)
                ]
    
    return plotSField

def plotFarField(fieldName):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    plotField = [
            InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
            InputDescription(inType.dropdown, "comp", label="Component", options = complist),
            InputDescription(inType.static, "project", staticValue="xy", hidden=True)
            ]
    return plotField

def plotCurrentOpt(fieldName):
    complist = ["Jx", "Jy", "Jz", "Mx", "My", "Mz"]
    
    plotCurrent = [
            InputDescription(inType.static, "field", label="Current", staticValue=fieldName),
            InputDescription(inType.dropdown, "comp", label="Component", options = complist),
            InputDescription(inType.xyzradio, "project", label="Abscissa - ordinate")
            ]

    return plotCurrent

def propPOInp(currentDict, scalarFieldDict, elemDict):
    sublist_currents = []
    sublist_sfields = []
    sublist_target = []

    if currentDict:
        for key, item in currentDict.items():
            sublist_currents.append(key)
    
    if scalarFieldDict:
        for key, item in scalarFieldDict.items():
            sublist_sfields.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            if item["gmode"] != 2:
                sublist_target.append(key)
    
    sublist_dev = ["CPU", "GPU"]


    propFields = [
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.dynamicDropdown, "mode", label="Propagation mode", subdict={
                "JM":[
                    InputDescription(inType.dropdown, "s_current", label="Source currents", options = sublist_currents),
                    InputDescription(inType.vectorStrings, "name_JM", label="Output currents", numFields=1)],
                "EH":[
                    InputDescription(inType.dropdown, "s_current", label="Source currents", options = sublist_currents),
                    InputDescription(inType.vectorStrings, "name_EH", label="Output fields", numFields=1)],
                "JMEH": [
                    InputDescription(inType.dropdown, "s_current", label="Source currents", options = sublist_currents),
                    InputDescription(inType.vectorStrings, "name_JM", label="Output currents", numFields=1),
                    InputDescription(inType.vectorStrings, "name_EH", label="Output fields", numFields=1)],
                "EHP": [ 
                    InputDescription(inType.dropdown, "s_current", label="Source currents", options = sublist_currents),
                    InputDescription(inType.vectorStrings, "name_EH", label="Output fields", numFields=1),
                    InputDescription(inType.vectorStrings, "name_P", label="Output frame", numFields=1)],
                "scalar":[
                    InputDescription(inType.dropdown, "s_scalarfield", label="Source scalar field", options = sublist_sfields),
                    InputDescription(inType.vectorStrings, "name_field", label="Output scalar field", numFields=1)]
                }),
            InputDescription(inType.vectorFloats, "epsilon", label="Relative permittivity", hints=[1], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propFields

def propPOFFInp(currentDict, elemDict):
    sublist_currents = []
    sublist_target = []
    if currentDict:
        for key, item in currentDict.items():
            sublist_currents.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            if item["gmode"] == 2:
                sublist_target.append(key)
    
    sublist_dev = ["CPU", "GPU"]


    propFields = [
            InputDescription(inType.dropdown, "s_current", label="Source currents", options = sublist_currents),
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.static, "mode", label="Propagation mode", staticValue="FF"),
            InputDescription(inType.vectorStrings, "name_EH", label="Output fields"),
            InputDescription(inType.vectorFloats, "epsilon", label="Relative permittivity", hints=[1], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propFields

def calcSpillEff(fieldDict, elemDict):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
   
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    formTaper = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = complist),
        InputDescription(inType.vectorFloats, "center", label="Center", numFields=2, oArray=True),
        InputDescription(inType.vectorFloats, "inner", label="Inner axes", numFields=2),
        InputDescription(inType.vectorFloats, "outer", label="Outer axes", numFields=2)
        ]

    return formTaper

def calcTaperEff(fieldDict, elemDict):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    formTaper = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = complist),
        ]

    return formTaper

def calcXpolEff(fieldDict, elemDict):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["gmode"] == 2:
                sublist_fields.append(key)
    
    formXpol = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "co_comp", label="Co-component", options = complist),
        InputDescription(inType.dropdown, "cr_comp", label="X-component", options = complist),
        ]

    return formXpol

def calcMBEff(fieldDict, elemDict):
    complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    mode_options = ["dB", "linear", "log"]

    formMB = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = complist),
        InputDescription(inType.vectorFloats, "thres", label="Threshold", numFields=1),
        InputDescription(inType.radio, "mode", label="Fitting mode", options=mode_options)
        ]

    return formMB

def saveSystemForm():
    return [InputDescription(inType.vectorStrings, "name", label="Name of system", numFields=1)]

def loadSystemForm(systemList):
    return [InputDescription(inType.dropdown, "name", label="Name of system", options=systemList)]

def focusFind(frameList):
    return [InputDescription(inType.dropdown, outputName="name_frame", label="Name of frame", options=frameList)]

def snapForm(elem, snapList, obj="element"):
    optionDict = {
            "Take" : [InputDescription(inType.vectorStrings, "snap_name", label="Snapshot name", numFields=1)],
            "Revert" : [InputDescription(inType.dropdown, "snap_name", label="Snapshot name", options=snapList)],
            "Delete" : [InputDescription(inType.dropdown, "snap_name", label="Snapshot name", options=snapList)],

            }
    form = [
            InputDescription(inType.static, "obj", staticValue=obj, hidden=True),
            InputDescription(inType.static, "name", label=f"Name of {obj}", staticValue=elem),
            InputDescription(inType.dynamicDropdown, "options", label="Options", subdict=optionDict)
            ]

    return form

def addGroupForm(elements):
    return[
        InputDescription(inType.vectorStrings, "name", toolTip= "Give the group a name"),
        InputDescription(inType.elementSelector, "selected", "elements", options = elements )
    ]

def copyForm(name):
    return [InputDescription(inType.static, "name", staticValue=name, hidden=True),
            InputDescription(inType.vectorStrings, "name_copy", label="Name of copy", numFields=1)]
