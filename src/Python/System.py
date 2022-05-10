import numpy as np
import matplotlib.pyplot as pt

import src.Python.Reflectors as Reflectors

class System(object):
    
    def __init__(self):
        self.num_ref = 0
        self.system = {}
    
    def addParabola(self, name="Parabola", a=1, b=1, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Parabola":
            name = name + "_{}".format(self.sys_id)
            
        p = Reflectors.Parabola(a, b, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = p
        self.num_ref += 1
        
    def addHyperbola(self, name="Hyperbola", a=1, b=1, c=1, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Parabola":
            name = name + "_{}".format(self.sys_id)
            
        h = Reflectors.Hyperbola(a, b, c, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def plotSystem(self):
        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        for refl in self.system.values():
            ax = refl.plotReflector(returns=True, ax_append=ax)
            
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        pt.show()
        
if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
