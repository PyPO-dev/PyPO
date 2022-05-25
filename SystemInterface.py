import numpy as np
import matplotlib.pyplot as pt

import src.Python.System as System

def SystemInterface():
    d_pri_sec = 3500 - 212.635
    d_sec_foc = 5500
    
    offTrans_p1 = np.array([0, 0, 0])
    rot_p1 = np.radians([0, 0, 0])
    cRot_p1 = np.array([0, 0, 0])
    
    rot_e1 = np.radians([0, 0, 0])
    offTrans_e1 = np.array([0, 0, 397])
    cRot_e1 = np.array([0, 0, 0])
    
    foc_1_h1 = np.array([0,0,3500])
    foc_2_h1 = np.array([0,0,3500 -  5606.286])
    ecc_h1 =  1.08208248
    
    # Parabola p1 initialization

    lims_x_p1 = [-5000, 5000]
    lims_y_p1 = [-5000, 5000]

    lims_u_p1 = [200/118, 5000/118]
    lims_v_p1 = [0, 2*np.pi]
    
    lims_u_h1 = [1, (3500 - 106.35)/(5606 / 2)]
    lims_v_h1 = [0, 2*np.pi]

    gridsize_p1 = [501, 501]

    # Hyperbola h1 initialization
    lims_x_h1 = [-310, 310]
    lims_y_h1 = [-310, 310]
    gridsize_h1 = [301, 301]
    
    #### Camera parameters
    center_cam = np.array([0,0,3500])
    lims_x_cam = [-100, 100]
    lims_y_cam = [-100, 100]
    gridsize_cam = [501, 501]
    
    focus_1 = np.array([0,0,3500])
    coef_p1 = [focus_1, cRot_p1]
    
    coef_h1 = [foc_1_h1, foc_2_h1, ecc_h1]
    #coef_h1 = [ 2590.5, 2590.5, 5606 / 2]
    
    ####
    s = System.System()
    #s.addParabola_ab(name = "p1", a = 3700, b = 100, cRot = cRot_p1, offRot = rot_p1, offTrans = offTrans_p1)
    s.addParabola(name = "p1", coef=coef_p1, mode='foc')
    #s.addHyperbola(name = "h1", a = 2590.5, b = 2590.5, c = 5606 / 2, cRot = cRot_e1, offRot = rot_e1, offTrans = offTrans_e1)
    s.addHyperbola(name = "h1", coef=coef_h1, mode='foc')
    #s.addHyperbola(name = "h1", coef=coef_h1, mode='man', offTrans = offTrans_e1)
    
    s.addCamera(name = "cam1", center=center_cam)
    
    print(s.system["p1"])
    print(s.system["h1"])
    #print(s.system["cam1"])
    
    #s.system["p1"].setGrid(lims_x_p1, lims_y_p1, gridsize_p1, calcArea=False, verbose=True)
    s.system["p1"].setGrid(lims_u_p1, lims_v_p1, gridsize_p1, calcArea=False, trunc=False, param='uv')
    s.system["p1"].rotateGrid()
    s.system["p1"].plotReflector(focus_1=False, focus_2=True, norm=True)

    #s.system["h1"].setGrid(lims_x_h1, lims_y_h1, gridsize_h1, orientation='inside', trunc=False)
    s.system["h1"].setGrid(lims_u_h1, lims_v_h1, gridsize_h1, orientation='outside', trunc=False, param='uv')
    s.system["h1"].rotateGrid()
    s.system["h1"].plotReflector(color='red', focus_1=True, focus_2=False, norm=True)

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    s.system["cam1"].plotCamera()
    
    s.plotSystem(focus_1=True, focus_2=True)
    '''
    pt.plot(s.system["p1"].area)
    pt.show()
    print(s.system["p1"].area)
    '''
    
    #s.initRaytracer(rCirc=0, NraysCirc=20, originChief=foc_2_h1, nomChief=np.array([0,0,1]), div_ang_x=3.2, div_ang_y=3.2)
    s.initRaytracer(rCirc=2000, NraysCirc=20, originChief=np.array([3000,0,3000]), nomChief=np.array([0,0,-1]), div_ang_x=0, div_ang_y=0)
    print(s.Raytracer)
    #s.Raytracer.plotRays()
    
    s.startRaytracer(surface="p1")
    #s.Raytracer.plotRays(frame=1)
    #s.startRaytracer(surface="h1")
    #s.startRaytracer(surface="p1")
    s.startRaytracer(surface="cam1")
    s.Raytracer.plotRays(frame=-1, quiv=False)

    s.plotSystem(focus_1=True, focus_2=True, plotRaytrace=True)#, exclude=[0,1,2])
    
if __name__ == "__main__":
    SystemInterface()
