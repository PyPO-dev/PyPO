from GUI.ParameterForms.InputDescription import inType, InputDescription

Plane = [
    InputDescription(inType.string, "name"),
    InputDescription(inType.integers, "gridsize", label="Grid Size", hints=[101,101], numFields=2)
]