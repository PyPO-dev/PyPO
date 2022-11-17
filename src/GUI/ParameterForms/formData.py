plane = {
    "name": 1,
    "deftype": {
        "xy": {"lims_X":2, "lims_Y":2 },
        "uv": {"lims_U":2, "lims_V":2 },
        "AoE": {"lims_Az":2, "lims_El":2 }
    } 
}

parabola = {
    "name": 1,
    "pmode":{
        "manual": {"coeffs": 3}, 
        "focus": {"vertex": 3, "focus_1":3} 
    },
    "gmode":{
        "xy": {"lims_x": 2, "lims_y":2}, 
        "uv": {"lims_u": 2, "lims_v":2} 
    },
    "gridsize": 2
    
}
print(plane)