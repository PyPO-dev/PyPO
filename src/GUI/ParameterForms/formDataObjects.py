from src.GUI.ParameterForms.InputDescription import inType, InputDescription

def xy_opts():
    return [InputDescription(inType.floats, "lims_x", label="X limits", oArray=True, numFields=2),
            InputDescription(inType.floats, "lims_y", label="Y limits", oArray=True, numFields=2)]

def uv_opts():
    return [InputDescription(inType.floats, "lims_u", label="U limits", oArray=True, numFields=2),
            InputDescription(inType.floats, "lims_v", label="V limits", oArray=True, numFields=2),
            InputDescription(inType.floats, "gcenter", label="XY center", oArray=True, numFields=2),
            InputDescription(inType.floats, "ecc_uv", label="UV eccentricity", numFields=1),
            InputDescription(inType.floats, "rot_uv", label="UV position angle", numFields=1)]

def AoE_opts():
    return [InputDescription(inType.floats, "lims_Az", label="Azimuth limits", oArray=True, numFields=2),
            InputDescription(inType.floats, "lims_El", label="Elevation limits", oArray=True, numFields=2)]

def focus_opts_hyp_ell():
    return [InputDescription(inType.floats, "focus_1", label="Upper focus xyz", oArray=True, numFields=3),
            InputDescription(inType.floats, "focus_2", label="Lower focus xyz", oArray=True, numFields=3),
            InputDescription(inType.floats, "ecc", label="Eccentricity", numFields=1)]

def makeParabolaInp():
    return [InputDescription(inType.string, "name"),
            InputDescription(inType.dropdown, "pmode", label="Parameter mode", subdict={
                "focus"     : [InputDescription(inType.floats, "focus_1", label="Focus xyz", oArray=True, numFields=3),
                                InputDescription(inType.floats, "vertex", label="Vertex xyz", oArray=True, numFields=3)],
                "manual"    : [InputDescription(inType.floats, "coeffs", label="AB coefficients", oArray=True, numFields=2)]
                }),
            InputDescription(inType.integers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.boolean, "flip", label="Flip Normal Vectors"),
            InputDescription(inType.dropdown, "gmode", label="Grid mode", subdict={
                "xy" : xy_opts(),
                "uv" : uv_opts()
            })]

def makeHyperbolaEllipseInp():
    return [InputDescription(inType.string, "name"),
            InputDescription(inType.dropdown, "pmode", label="Parameter mode", subdict={
                "focus"     : focus_opts_hyp_ell(),
                "manual"    : [InputDescription(inType.floats, "coeffs", label="ABC coefficients", oArray=True, numFields=3)]
                }),
            InputDescription(inType.integers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.dropdown, "gmode", label="Grid mode", subdict={
                "xy" : xy_opts(),
                "uv" : uv_opts()
            })]

def makeQuadricSurfaceInp():
    return [
        InputDescription(inType.dropdown, "type", subdict={
            "Parabola": makeParabolaInp(),
            "Hyperbola": makeHyperbolaEllipseInp(),
            "Ellipse": makeHyperbolaEllipseInp()
        })
    ]

def makePlaneInp():
    return [InputDescription(inType.string, "name"),
            InputDescription(inType.integers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.dropdown, "gmode", label="Grid mode", subdict={
                "xy" : xy_opts(),
                "uv" : uv_opts(),
                "AoE" : AoE_opts()
            })]
            #InputDescription(inType.integers, "flip", label="Flip normal vectors", numFields=1)]

def makeTransformationForm(elementName):
    return[
        InputDescription(inType.static, "element", staticValue=elementName),
        InputDescription(inType.dropdown, "type", subdict={
            "Translation":[
                InputDescription(inType.floats, "vector", label="Translation Vector", hints=["x","y","z"], numFields=3,oArray=True)],
            "Rotation": [
                InputDescription(inType.floats, "vector", label="Rotation Vector", hints=["x","y","z"], numFields=3,oArray=True),
                InputDescription(inType.floats, "centerOfRotation", label="Center of Rotation", hints=["x","y","z"], numFields=3,oArray=True)
                ]
        })
    ]

# NOTE
def initFrameInp():
    return [InputDescription(inType.string, "name", label="Name of frame", numFields=1),
            InputDescription(inType.integers, "nRays", label="# of rays", hints=[0], numFields=1),
            InputDescription(inType.integers, "nRing", label="# of rings", hints=[0], numFields=1),
            InputDescription(inType.floats, "angx", label="X-apex angle", hints=[0], numFields=1),
            InputDescription(inType.floats, "angy", label="Y-apex angle", hints=[0], numFields=1),
            InputDescription(inType.floats, "a", label="X radius of outer ring", hints=[0], numFields=1),
            InputDescription(inType.floats, "b", label="Y radius of outer ring", hints=[0], numFields=1),
            InputDescription(inType.floats, "tChief", label="Chief ray tilt", hints=[0,0,1], numFields=3, oArray=True),
            InputDescription(inType.floats, "oChief", label="Chief ray origin", hints=[0,0,0], numFields=3, oArray=True)]

def plotFrameInp(frameDict):
    sublist_frames = []
    if frameDict:
        for key in frameDict.keys():
            sublist_frames.append(key)
    
    plotFrame = [
            InputDescription(inType.dropdown, "frame", label="Frame", sublist = sublist_frames),
            InputDescription(inType.string, "project", label="Abscissa - ordinate", hints=["xy"], numFields=1)
            ]

    return plotFrame

def plotFrameOpt(frameName):
    plotFrame = [
            InputDescription(inType.static, "frame", label="Frame", staticValue=frameName),
            InputDescription(inType.string, "project", label="Abscissa - ordinate", hints=["xy"], numFields=1)
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
            InputDescription(inType.dropdown, "frame_in", label="Input frame", sublist = sublist_frames),
            InputDescription(inType.dropdown, "target", label="Target surface", sublist = sublist_target),
            InputDescription(inType.string, "frame_out", label="Name of output frame", numFields=1),
            InputDescription(inType.floats, "epsilon", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.integers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.floats, "t0", label="Initial guess", hints=[100], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", sublist = sublist_dev)
            ]

    return propRays
    
def initGaussianInp(elemDict):
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initGauss = [
            InputDescription(inType.dropdown, "surface", label="Gaussian beam surface", sublist = sublist_surf),
            InputDescription(inType.string, "name", label="Beam name", numFields=1),
            InputDescription(inType.floats, "lam", label="Wavelength of radiation", hints=[1], numFields=1),
            InputDescription(inType.floats, "w0x", label="Beamwaist in X", hints=[5], numFields=1),
            InputDescription(inType.floats, "w0y", label="Beamwaist in Y", hints=[5], numFields=1),
            InputDescription(inType.floats, "n", label="Refractive index", hints=[1], numFields=1),
            InputDescription(inType.floats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.floats, "z", label="Focal distance", hints=[0], numFields=1),
            InputDescription(inType.floats, "pol", label="Polarization", hints=[1,0,0], numFields=3, oArray=True)
            ]
    
    return initGauss

def plotFieldOpt(fieldName, elemDict, fieldType):
    sublist_surf = []

    if fieldType == "EH":
        complist = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    
    elif fieldType == "JM":
        complist = ["Jx", "Jy", "Jz", "Mx", "My", "Mz"]

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)
    
    plotField = [
            InputDescription(inType.static, "field", label="Field/current", staticValue=fieldName),
            InputDescription(inType.dropdown, "surface", label="Beam surface", sublist = sublist_surf),
            InputDescription(inType.dropdown, "comp", label="Component", sublist = complist),
            InputDescription(inType.string, "project", label="Abscissa - ordinate", hints=["xy"], numFields=1)
            ]

    return plotField
"""
def propPOInp(currentDict, elemDict):
    sublist_currents = []
    sublist_target = []
    if currentDict:
        for key, item in currentDict.items():
            sublist_currents.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            sublist_target.append(key)
    
    sublist_dev = ["CPU", "GPU"]

    EHDict = {
            
            }

    propFields = [
            InputDescription(inType.dropdown, "currents", label="Source currents", sublist = sublist_currents),
            InputDescription(inType.dropdown, "target", label="Target surface", sublist = sublist_target),
            InputDescription(inType.dropdown, "return", subdict={
                "JM":[
                    InputDescription(inType.string, "name_JM", label="Output currents", numFields=1)],
                "EH":[
                    InputDescription(inType.string, "name_EH", label="Output fields", numFields=1)],
                "JM": [
                    InputDescription(inType.string, "name_JM", label="Output currents", numFields=1),
                    InputDescription(inType.string, "name_EH", label="Output fields", numFields=1)
                    ]
            InputDescription(inType.string, "", label="Name of output frame", numFields=1),
            InputDescription(inType.floats, "epsilon", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.integers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.floats, "t0", label="Initial guess", hints=[100], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", sublist = sublist_dev)
            ]

    return propRays
"""

def saveSystemForm():
    return [InputDescription(inType.string, "name", label="Name of system", numFields=1)]

def loadSystemForm():
    
    return [InputDescription(inType.dropdown, "name", label="Name of system", numFields=1)]


# END NOTE
