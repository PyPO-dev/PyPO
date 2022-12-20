from src.GUI.ParameterForms.InputDescription import inType, InputDescription

Plane = [
    InputDescription(inType.string, "name"),
    InputDescription(inType.integers, "gridsize", label="Grid Size", hints=[101,101], numFields=2)
]
# NOTE
RTGen = [
    InputDescription(inType.integers, "nRays", label="# of rays", hints=[0], numFields=1),
    InputDescription(inType.integers, "nRing", label="# of rings", hints=[0], numFields=1),
    InputDescription(inType.floats, "angx", label="X-apex angle", hints=[0], numFields=1),
    InputDescription(inType.floats, "angy", label="Y-apex angle", hints=[0], numFields=1),
    InputDescription(inType.floats, "a", label="X radius of outer ring", hints=[0], numFields=1),
    InputDescription(inType.floats, "b", label="Y radius of outer ring", hints=[0], numFields=1),
    InputDescription(inType.floats, "tChief", label="Chief ray tilt", hints=[0,0,1], numFields=3, oArray=True),
    InputDescription(inType.floats, "oChief", label="Chief ray origin", hints=[0,0,0], numFields=3, oArray=True)
]

def plotFrameInp(frameDict):
    sublist_frames = []
    if frameDict:
        for key, item in frameDict.items():
            sublist_frames.append(key)
    
    plotFrame = [
            InputDescription(inType.dropdown, "frame", label="Frame", sublist = sublist_frames),
            InputDescription(inType.string, "project", label="Abscissa and ordinate", hints=["xy"], numFields=1)
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
            InputDescription(inType.floats, "epsilon", label="Accuracy", hints=[1e-3], numFields=1),
            InputDescription(inType.integers, "nThreads", label="# of threads", hints=[1], numFields=1),
            InputDescription(inType.floats, "t0", label="Initial guess", hints=[100], numFields=1),
            InputDescription(inType.dropdown, "device", label="Hardware to use", sublist = sublist_dev)
            ]

    return propRays

# END NOTE
