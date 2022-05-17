import numpy as np
import matplotlib.pyplot as pt

import src.Python.Reflectors as Reflectors
import src.Python.System as System

def SystemInterface():
    d_pri_sec = 3500 - 212.635
    d_sec_foc = 5500
    
    offTrans_p1 = np.array([0, 0, 0])
    rot_p1 = np.radians([0, 0, 0])
    cRot_p1 = np.array([0, 0, 0])
    
    rot_e1 = np.radians([0, 0, 0])
    offTrans_e1 = np.array([0, 0, 697])
    cRot_e1 = offTrans_e1
    
    # Parabola p1 initialization
    '''
    lims_x_p1 = [-5000, 5000]
    lims_y_p1 = [-5000, 5000]
    '''
    lims_x_p1 = [-5000, 5000]
    lims_y_p1 = [-5000, 5000]
    gridsize_p1 = [501, 501]

    # Hyperbola h1 initialization
    lims_x_h1 = [-310, 310]
    lims_y_h1 = [-310, 310]
    gridsize_h1 = [301, 301]
    
    
    ####
    s = System.System()
    #s.addParabola_ab(name = "p1", a = 3700, b = 37, cRot = cRot_p1, offRot = rot_p1, offTrans = offTrans_p1)
    s.addParabola_foc(name = "p1", focus_1 = np.array([0,0,3500]), cRot = cRot_p1, offRot = rot_p1, offTrans = offTrans_p1)
    s.addHyperbola_ab(name = "h1", a = 2590.5, b = 2590.5, c = 5606 / 2, cRot = cRot_e1, offRot = rot_e1, offTrans = offTrans_e1)
    
    print(s.system["p1"])
    print(s.system["h1"])
    
    s.system["p1"].setGrid(lims_x_p1, lims_y_p1, gridsize_p1)
    s.system["p1"].rotateGrid()
    s.system["p1"].plotReflector(focus_1=True, focus_2=True)
    
    s.system["h1"].setGrid(lims_x_h1, lims_y_h1, gridsize_h1)
    s.system["h1"].rotateGrid()
    s.system["h1"].plotReflector(color='red', focus_1=True)
    
    s.plotSystem(focus_1=True, focus_2=True)
    '''
    pt.plot(s.system["p1"].area)
    pt.show()
    print(s.system["p1"].area)
    '''
if __name__ == "__main__":
    SystemInterface()
