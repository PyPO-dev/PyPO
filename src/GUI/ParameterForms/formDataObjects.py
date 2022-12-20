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


# END NOTE
