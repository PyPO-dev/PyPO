"""!
@file
This file contains all forms used by the GUI for interaction with the user.
Because all functions return lists containing forms, the return will not be explicitly stated per form.
"""

from src.GUI.ParameterForms.InputDescription import inType, InputDescription
from PyPO.Enums import FieldComponents, CurrentComponents, Projections

FieldComponentList = [o for o in FieldComponents]
FieldComponentList.pop()
FieldComponentListStrings = [do.name for do in FieldComponentList] 
CurrentComponentList = [o for o in CurrentComponents]
CurrentComponentListStrings = [do.name for do in CurrentComponents] 

def surface_error():
    """!
    Options for adding surface error
    """
    "rms en rms_seed"
    return[
        InputDescription(inType.dynamicRadio, "error_checkbox", label = "Surface error", subDict={
            "None"     : [],
            "Yes"    : [
                InputDescription(inType.vectorFloats, "rms", label= "Surface error RMS"),
                InputDescription(inType.vectorIntegers, "rms_seed", label= "Surface error seed"),
                ]
            })
    ]

def xy_opts():
    """!
    Options for generating an element from an xy parametrisation.
    """
    return [InputDescription(inType.vectorFloats, "lims_x", label="X limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "lims_y", label="Y limits", oArray=True, numFields=2),
            
            ]+surface_error()

def uv_opts():
    """!
    Options for generating an element from a uv parametrisation.
    """
    return [InputDescription(inType.vectorFloats, "lims_u", label="U limits", oArray=True, numFields=2, toolTip = "U limits: inner and outer diameter."),
            InputDescription(inType.vectorFloats, "lims_v", label="V limits", prefill = True, hints=[0., 360.], oArray=True, numFields=2, toolTip = "V angle in degrees."),
            InputDescription(inType.vectorFloats, "gcenter", label="XY center", hints = [0.,0.], oArray=True, numFields=2, prefill = True),
            InputDescription(inType.vectorFloats, "ecc_uv", label="UV eccentricity", numFields=1, hints = [0.], prefill = True),
            InputDescription(inType.vectorFloats, "rot_uv", label="UV position angle", numFields=1, hints = [0.], prefill = True),
            
            ]+surface_error()

def AoE_opts():
    """!
    Options for generating a far-field element from an AoE (Azimuth-over-Elevation) parametrisation.
    """
    return [InputDescription(inType.vectorFloats, "lims_Az", label="Azimuth limits", oArray=True, numFields=2),
            InputDescription(inType.vectorFloats, "lims_El", label="Elevation limits", oArray=True, numFields=2)]

def focus_opts_hyp_ell():
    """!
    Extra options for hyperboloids and ellipsoids for supplying quadric parameters and gridding options.
    """
    return [
        InputDescription(inType.vectorFloats, "focus_1", label="Upper focus xyz", oArray=True, numFields=3),
            InputDescription(inType.vectorFloats, "focus_2", label="Lower focus xyz", oArray=True, numFields=3),
            InputDescription(inType.vectorFloats, "ecc", label="Eccentricity", numFields=1)
    ]

def makeParabolaInp():
    """!
    Form for generating a paraboloid element.
    """
    return [
        InputDescription(inType.vectorStrings, "name", hints = ["Parabola"], prefill = True),
        InputDescription(inType.dynamicRadio, "pmode", label="Parameter mode", subDict={
            "focus"     : [InputDescription(inType.vectorFloats, "focus_1", label="Focus", oArray=True, numFields=3, toolTip = "vector x, y, z"),
                            InputDescription(inType.vectorFloats, "vertex", label="Vertex", oArray=True, numFields=3, toolTip = "vector x, y, z")],
            "manual"    : [InputDescription(inType.vectorFloats, "coeffs", label="AB coefficients", oArray=True, numFields=2)]
            }),
        InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
        InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
        InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subDict={
            "xy" : xy_opts(),
            "uv" : uv_opts()
        })
    ]

def makeHyperbolaInp():
    """!
    Form for generating a hyperboloid element.
    """
    return [InputDescription(inType.vectorStrings, "name", hints = ["Hyperbola"], prefill = True),
            InputDescription(inType.dynamicRadio, "pmode", label="Parameter mode", subDict={
                "focus"     : focus_opts_hyp_ell(),
                "manual"    : [InputDescription(inType.vectorFloats, "coeffs", label="ABC coefficients", oArray=True, numFields=3)]
                }),
            InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
            InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subDict={
                "xy" : xy_opts(),
                "uv" : uv_opts()
            })]

def makeEllipseInp():
    """!
    Form for generating an ellipsoid element.
    """
    return [InputDescription(inType.vectorStrings, "name", hints = ["Ellipse"], prefill = True),
            InputDescription(inType.dynamicRadio, "pmode", label="Parameter mode", subDict={
                "focus"     : focus_opts_hyp_ell(),
                "manual"    : [InputDescription(inType.vectorFloats, "coeffs", label="ABC coefficients", oArray=True, numFields=3)]
                }),
            InputDescription(inType.radio, "orient", label="Major axis orientation", options=["x", "z"]),
            InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
            InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subDict={
                "xy" : xy_opts(),
                "uv" : uv_opts()
            })]
def makeQuadricSurfaceInp():
    """!
    Menu for generating quadric elements.
    """
    return [
        InputDescription(inType.dynamicDropdown, "type", subDict={
            "Parabola": makeParabolaInp(),
            "Hyperbola": makeHyperbolaInp(),
            "Ellipse": makeEllipseInp()
        })
    ]

def makePlaneInp():
    """!
    Menu for generating planar elements.
    """
    return [InputDescription(inType.vectorStrings, "name"),
            InputDescription(inType.vectorIntegers, "gridsize", label="Grid size", hints=[101,101], numFields=2, oArray=True),
            InputDescription(inType.checkbox, "flip", label="Flip Normal Vectors"),
            InputDescription(inType.dynamicRadio, "gmode", label="Grid mode", subDict={
                "xy" : xy_opts(),
                "uv" : uv_opts(),
                "AoE" : AoE_opts()
                })
            ]

def makeTransformationForm(name, obj="element"):
    """!
    Options for transforming elements and groups. Also used for transforming frames.
    
    @param name Name of element/group/frame in system.
    @param obj Type of object to transform. Defaults to "element".
    """
    return[
        InputDescription(inType.static, obj, staticValue=name),
        InputDescription(inType.radio, "mode", label="Transformation mode", options=[
            "Relative", "Absolute"
            ]),
        InputDescription(inType.dynamicRadio, "type", subDict={
            "Translation":[
                InputDescription(inType.vectorFloats, "vector", label="Translation Vector", hints=[0.,0.,0.], numFields=3,oArray=True, prefill = True)],
            "Rotation": [
                InputDescription(inType.vectorFloats, "vector", label="Rotation Vector", hints=[0.,0.,0.], numFields=3,oArray=True, prefill = True),
                InputDescription(inType.vectorFloats, "pivot", label="Center of Rotation", hints=[0.,0.,0.], numFields=3,oArray=True, prefill = True)
                ]
        })
    ]

def initTubeFrameInp():
    """!
    Options for generating a tubular ray-trace frame.
    """
    return [InputDescription(inType.vectorStrings, "name", label="Name of frame", numFields=1),
            InputDescription(inType.vectorIntegers, "nRays", label="Number of rays", hints=[0], numFields=1),
            InputDescription(inType.vectorIntegers, "nRing", label="Number of rings", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "angx0", label="X-apex angle", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "angy0", label="Y-apex angle", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "x0", label="X radius of outer ring", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "y0", label="Y radius of outer ring", hints=[0], numFields=1)
            ]

def initGaussianFrameInp():
    """!
    Options for generating a Gaussian ray-trace frame.
    """
    return [InputDescription(inType.vectorStrings, "name", label="Name of frame", numFields=1),
            InputDescription(inType.vectorIntegers, "nRays", label="Number of rays", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "n", label="Refractive index of medium", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "x0", label="X beamwaist", hints=[5], numFields=1),
            InputDescription(inType.vectorFloats, "y0", label="Y beamwaist", hints=[5], numFields=1),
            InputDescription(inType.dynamicRadio, "setseed", label="Set seed", subDict={
                "random" : [],
                "set" : [InputDescription(inType.vectorIntegers, "seed", label="", hints=[0], numFields=1)]
            })]

def plotFrameOpt(frameName):
    """!
    Options for plotting a ray-trace frame.
    
    @param frameName Name of frame to plot.
    """
    plotFrame = [
            InputDescription(inType.static, "frame", label="Frame", staticValue=frameName),
            InputDescription(inType.xyzRadio, "project", label="Abscissa - ordinate")
            ]

    return plotFrame

def plotRayTraceForm(frames):
    """!
    Options for plotting a set of frames in a plotSystem figure.
    
    @param frames List containing names of available frames.
    """
    return [
        InputDescription(inType.dynamicRadio, "frames", subDict = {
            "All" : [],
            "Select":[InputDescription(inType.elementSelector, "selection", options = frames)]
        }),
    ]


def propRaysInp(frameDict, elemDict):
    """!
    Options for propagating a frame of rays to a target element.
    
    @param frameDict System dictionary containing all frames.
    @param elemDict System dictionary containing all elements.
    """
    sublist_frames = []
    sublist_target = []
    if frameDict:
        for key, item in frameDict.items():
            sublist_frames.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            if item["gmode"] != 2:
                sublist_target.append(key)
    
    sublist_dev = ["CPU", "GPU"]

    propRays = [
            InputDescription(inType.dropdown, "fr_in", label="Input frame", options = sublist_frames),
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.vectorStrings, "fr_out", label="Name of output frame", numFields=1),
            InputDescription(inType.vectorFloats, "tol", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="Number of threads", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "t0", label="Initial guess", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propRays
    
def initPSInp(elemDict):
    """!
    Options for generating a vectorial point-source field/current for PO calculations.
    
    @param elemDict System dictionary containing all elements.
    """
    sublist_surf = []

    if elemDict:
        for key, item in elemDict.items():
            if item["type"] == 3: # Only append plane types
                sublist_surf.append(key)

    initPS = [
            InputDescription(inType.dropdown, "surface", label="Source surface", options = sublist_surf),
            InputDescription(inType.vectorStrings, "name", label="Beam name", numFields=1),
            InputDescription(inType.vectorFloats, "lam", label="Wavelength", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "E0", label="Peak value", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "phase", label="Phase", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "pol", label="Polarization", hints=[1,0,0], numFields=3, oArray=True)
            ]
    
    return initPS

def initSPSInp(elemDict):
    """!
    Options for generating a scalar point-source field for PO calculations.
    
    @param elemDict System dictionary containing all elements.
    """
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
    """!
    Options for generating a vectorial complex-valued Gaussian field/current for PO calculations.
    
    @param elemDict System dictionary containing all elements.
    """
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
            InputDescription(inType.vectorFloats, "dxyz", label="Astigmatic distance", hints=[0], numFields=1),
            InputDescription(inType.vectorFloats, "pol", label="Polarization", hints=[1,0,0], numFields=3, oArray=True)
            ]
    
    return initGauss

def initSGaussianInp(elemDict):
    """!
    Options for generating a scalar complex-valued Gaussian field for PO calculations.
    
    @param elemDict System dictionary containing all elements.
    """
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
    """!
    Options for plotting a field object. Also contains possibility for plotting beam cross sections.
    
    @param fieldName Name of field object to plot.
    """

    plotField = [
            InputDescription(inType.dynamicDropdown, "plot_type", label="Type", subDict = {
                "Pattern" : [
                    InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                    InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints= FieldComponentListStrings),
                    InputDescription(inType.xyzRadio, "project", label="Abscissa - ordinate"),
                    InputDescription(inType.checkbox, "phase", label="Include phase", prefill=True)],
                "Cross-sections" : [
                    InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                    InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints= FieldComponentListStrings),
                    ]
                })
            ]

    return plotField

def plotSField(fieldName):
    """!
    Options for plotting a scalarfield object.
    
    @param fieldName Name of field object to plot.
    """
    
    plotSField = [
            InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
            InputDescription(inType.xyzRadio, "project", label="Abscissa - ordinate")
            ]
    
    return plotSField

def plotFarField(fieldName):
    """!
    Options for plotting a field object defined on a far-field grid. 
    Also contains possibility for plotting beam cross sections.
    
    @param fieldName Name of field object to plot.
    """
    
    plotField = [
            InputDescription(inType.dynamicDropdown, "plot_type", label="Type", subDict = {
                "Pattern" : [
                    InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                    InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints= FieldComponentListStrings),
                    InputDescription(inType.static, "project", staticValue=Projections.xy, hidden=True),
                    InputDescription(inType.checkbox, "phase", label="Include phase", prefill=False)],
                "Cross-sections" : [
                    InputDescription(inType.static, "field", label="Field", staticValue=fieldName),
                    InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints= FieldComponentListStrings),
                    ]
                })
            ]
    return plotField

def plotCurrentOpt(currentName):
    """!
    Options for plotting a current object.
    
    @param fieldName Name of current object to plot.
    """
    
    plotCurrent = [
                    InputDescription(inType.static, "field", label="Field", staticValue=currentName),
                    InputDescription(inType.dropdown, "comp", label="Component", options = CurrentComponentList, hints= CurrentComponentListStrings),
                    InputDescription(inType.xyzRadio, "project", label="Abscissa - ordinate")]

    return plotCurrent

def propPOInp(currentDict, scalarFieldDict, elemDict):
    """!
    Options for propagating the field generated by a current distribution to a target element.
    If propagating a scalarfield, the propagation is done using the Lipmann-Schwinger equation.
    
    @param currentDict System dictionary containing all currents.
    @param scalarFieldDict System dictionary containing all scalarfields.
    @param elemDict System dictionary containing all elements.
    """
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
    
    sublist_exp = ["forward", "backward"]

    sublist_dev = ["CPU", "GPU"]


    propFields = [
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.dynamicDropdown, "mode", label="Propagation mode", subDict={
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
            InputDescription(inType.radio, "exp", label="Time direction", options = sublist_exp),
            InputDescription(inType.vectorFloats, "epsilon", label="Relative permittivity", hints=[1], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="Number of threads", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propFields

def propPOFFInp(currentDict, elemDict):
    """!
    Options for propagating the field generated by a current distribution to a far-field target element.
    
    @param currentDict System dictionary containing all currents.
    @param elemDict System dictionary containing all elements.
    """
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
            InputDescription(inType.vectorIntegers, "nThreads", label="Number of threads", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propFields

def propPOHybridInp(fieldDict, frameDict, elemDict):
    """!
    Options for propagating the reflected field using the associated Poynting vectors to a target element.
    
    @param fieldDict System dictionary containing all fields.
    @param frameDict System dictionary containing all frames.
    @param elemDict System dictionary containing all elements.
    """
    sublist_fields = []
    sublist_frames = []
    sublist_target = []
    
    if fieldDict:
        for key, item in fieldDict.items():
            sublist_fields.append(key)
    
    if frameDict:
        for key, item in frameDict.items():
            sublist_frames.append(key)
    
    if elemDict:
        for key, item in elemDict.items():
            if item["gmode"] != 2:
                sublist_target.append(key)
    sublist_dev = ["CPU", "GPU"]

    propFields = [
            InputDescription(inType.dropdown, "fr_in", label="Poynting", options = sublist_frames),
            InputDescription(inType.dropdown, "field_in", label="Reflected field", options = sublist_fields),
            InputDescription(inType.dropdown, "t_name", label="Target surface", options = sublist_target),
            InputDescription(inType.vectorStrings, "fr_out", label="Output frame", numFields=1),
            InputDescription(inType.vectorStrings, "field_out", label="Output field", numFields=1),
            InputDescription(inType.dynamicRadio, "_start", label="Use start", subDict={
                    "yes" : [InputDescription(inType.vectorFloats, "start", oArray=True, label="Start co-ordinate", numFields=3)],
                    "no" : [InputDescription(inType.static, "start", label="", staticValue=None, hidden=True)]
                }), 
            InputDescription(inType.dynamicRadio, "_interp", label="Interpolate", subDict={
                    "yes" : [InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints = FieldComponentListStrings)],##TODO Set 'ALL components' option 
                    "no" : [InputDescription(inType.static, "comp", label="", staticValue=True, hidden=True)]
                }), 
            InputDescription(inType.vectorFloats, "tol", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.vectorIntegers, "nThreads", label="Number of threads", hints=[1], numFields=1),
            InputDescription(inType.vectorFloats, "t0", label="Initial guess", hints=[1], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", options = sublist_dev)
            ]

    return propFields

def calcSpillEff(fieldDict, elemDict):
    """!
    Options for calculating the spillover efficiency on a surface by using an aperture mask.
    
    @param fieldDict System dictionary containing all fields.
    @param elemDict System dictionary containing all elements.
    """
   
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    formTaper = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints = FieldComponentListStrings),
        InputDescription(inType.vectorFloats, "center", label="Center", numFields=2, oArray=True),
        InputDescription(inType.vectorFloats, "inner", label="Inner axes", numFields=2),
        InputDescription(inType.vectorFloats, "outer", label="Outer axes", numFields=2)
        ]

    return formTaper

def calcTaperEff(fieldDict, elemDict):
    """!
    Options for calculating the taper efficiency on a planar surface.
    
    @param fieldDict System dictionary containing all fields.
    @param elemDict System dictionary containing all elements.
    """
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    formTaper = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints = FieldComponentListStrings),
        ]

    return formTaper

def calcXpolEff(fieldDict, elemDict):
    """!
    Options for calculating the cross-polar efficiency on a far-field element.
    
    @param fieldDict System dictionary containing all fields.
    @param elemDict System dictionary containing all elements.
    """
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["gmode"] == 2:
                sublist_fields.append(key)
    
    formXpol = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "co_comp", label="Co-component", options = FieldComponentList, hints = FieldComponentListStrings),
        InputDescription(inType.dropdown, "cr_comp", label="X-component", options = FieldComponentList, hints = FieldComponentListStrings),
        ]

    return formXpol

def calcMBEff(fieldDict, elemDict):
    """!
    Options for calculating the main-beam efficiency on a planar element.
    
    @param fieldDict System dictionary containing all fields.
    @param elemDict System dictionary containing all elements.
    """
    
    sublist_fields = []
    if fieldDict:
        for key, item in fieldDict.items():
            if elemDict[item.surf]["type"] == 3:
                sublist_fields.append(key)
    
    mode_options = ["dB", "linear", "log"]

    formMB = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints = FieldComponentListStrings),
        InputDescription(inType.vectorFloats, "thres", label="Threshold", numFields=1),
        InputDescription(inType.radio, "mode", label="Fitting mode", options=mode_options)
        ]

    return formMB

def calcHPBW(fieldDict):
    """!
    Options for calculating the half-power beamwidths of a field component in the E and H-planes.
    The field component is first transformed so that it is centered in the origin and aligned with the x and y axes.
    
    @param fieldDict System dictionary containing all fields.
    """
   
    sublist_fields = []
    if fieldDict:
        for key in fieldDict.keys():
            sublist_fields.append(key)

    formHPBW = [
        InputDescription(inType.dropdown, "f_name", label="Field", options = sublist_fields),
        InputDescription(inType.dropdown, "comp", label="Component", options = FieldComponentList, hints = FieldComponentListStrings)
        ]

    return formHPBW

def mergeBeamsForm(itemDict, surf):
    """!
    Options for merging beams/currents.
    
    @param itemDict Dictionary containing fields or currents in system.
    @param surf Selected surface for beam merging.
    """
    listBound = _selectBound(itemDict, surf)

    mergeList = [InputDescription(inType.elementSelector, "beams", "Merge", options = listBound),
                InputDescription(inType.vectorStrings, "merged_name", label="Merged name", numFields=1)]

    return mergeList

def selectSurface(elemDict):
    """!
    Select a surface form.
    Used for merging beams on a surface.
    
    @param elemDict Dictionary containing all elements in system.
    """
    optlist = ["Fields", "Currents"]
    
    selectSurf = [InputDescription(inType.dropdown, "surf", label="Merge surface", options=list(elemDict.keys())),
            InputDescription(inType.radio, "mode", label="Merge object", options = optlist)]
    return selectSurf

def _selectBound(itemDict, surf):
    """!
    Private method for finding bound PO fields and currents given a surface.
    
    @param itemDict Dictionary containing the items to be checked.
    @param surf Surface to find fields or currents on.
    """
    listBound = []
    for key, item in itemDict.items():
        if item.surf == surf:
            listBound.append(key)

    return listBound

def saveSystemForm():
    """!
    Options for saving the current system in the PyPO/save/systems/ folder.
    """
    return [InputDescription(inType.vectorStrings, "name", label="Name of system", numFields=1)]

def loadSystemForm(systemList):
    """!
    Options for loading a system in the PyPO/save/systems/ folder into the current system.
    
    @param systemList List of systems present in PyPO/save/systems/.
    """
    return [InputDescription(inType.dropdown, "name", label="Name of system", options=systemList)]

def focusFind(frameList):
    """!
    Options for finding the focus of a ray-trace frame.
    
    @param frameList List of names of frames in system.
    """
    return [InputDescription(inType.dropdown, outputName="name_frame", label="Name of frame", options=frameList)]

def snapForm(elem, snapList, obj="element"):
    """!
    Options for taking/reverting/deleting a snapshot of an object.
    
    @param elem Name of object to snap.
    @param snapList List of current snapshots belonging to the object.
    @param obj Type of object to be snapped.
    """
    optionDict = {
            "Take" : [InputDescription(inType.vectorStrings, "snap_name", label="Snapshot name", numFields=1)],
            "Revert" : [InputDescription(inType.dropdown, "snap_name", label="Snapshot name", options=snapList)],
            "Delete" : [InputDescription(inType.dropdown, "snap_name", label="Snapshot name", options=snapList)],

            }
    form = [
            InputDescription(inType.static, "obj", staticValue=obj, hidden=True),
            InputDescription(inType.static, "name", label=f"Name of {obj}", staticValue=elem),
            InputDescription(inType.dynamicDropdown, "options", label="Options", subDict=optionDict)
            ]

    return form

def addGroupForm(elementList):
    """!
    Options for creating a group of elements.
    
    @param elementList List of all element names.
    """
    return[
        InputDescription(inType.vectorStrings, "name", toolTip= "Give the group a name"),
        InputDescription(inType.elementSelector, "selected", "elements", options = elementList)
    ]

def copyForm(name):
    """!
    Options for copying an object to another object, potentially under a new name.
    
    @param name Name of object to be copied.
    """
    return [InputDescription(inType.static, "name", staticValue=name, hidden=True),
            InputDescription(inType.vectorStrings, "name_copy", label="Name of copy", numFields=1)]
