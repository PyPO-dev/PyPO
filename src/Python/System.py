# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt

# POPPy-specific modules
import src.Python.Reflectors as Reflectors
import src.Python.RayTrace as RayTrace

class System(object):
    
    def __init__(self):
        self.num_ref = 0
        self.system = {}
        
    def __str__(self):
        pass
    
    def __iter__(self):
        pass
    
    #### ADD REFLECTOR METHODS
    # Takes as input a and b
    def addParabola_ab(self, name="Parabola", a=1, b=1, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Parabola":
            name = name + "_{}".format(self.sys_id)
            
        p = Reflectors.Parabola(a, b, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = p
        self.num_ref += 1
    
    # Takes as input focus_1 position. If created with this funcy=tion, automatically a=b
    def addParabola_foc(self, name="Parabola", focus_1=np.array([0,0,1]), cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Parabola":
            name = name + "_{}".format(self.sys_id)
        
        a = 2 * np.sqrt(focus_1[2])
        b = a
        
        p = Reflectors.Parabola(a, b, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = p
        self.num_ref += 1
        
    def addHyperbola_ab(self, name="Hyperbola", a=1, b=1, c=1, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Hyperbola":
            name = name + "_{}".format(self.sys_id)
            
        h = Reflectors.Hyperbola(a, b, c, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def addHyperbola_foc(self, name="Hyperbola", a=1, b=1, c=1, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Hyperbola":
            name = name + "_{}".format(self.sys_id)
            
        h = Reflectors.Hyperbola(a, b, c, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def addEllipse_ab(self, name="Ellipse", a=2, b=3, c=5, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.sys_id)
        
        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
        
    # Place and create ellipse using focus_1 and confocal distance c. Automatically creates a symmetric ellipsoid
    def addEllipse_foc(self, name="Ellipse", focus_1=np.array([0,0,1]), focus_2=np.array([0,0,1]), cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.sys_id)

        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
        
    def removeElement(self, name):
        del self.system[name]
        self.num_ref -= 1
    
    #### PLOTTING OPTICAL SYSTEM
    def plotSystem(self, focus_1=False, focus_2=False, plotRaytrace=False):
        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        for refl in self.system.values():
            ax = refl.plotReflector(returns=True, ax_append=ax, focus_1=focus_1, focus_2=focus_2)
            
        if plotRaytrace:
            ax = self.Raytracer.plotRaysSystem(ax_append=ax)
            
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        pt.show()
        
    def initRaytracer(self, NraysCirc=0, NraysCross=0, 
                 rCirc=0, div_ang_x=2*np.pi, div_ang_y=2*np.pi,
                 originChief=np.array([0,0,0]), 
                 tiltChief=np.array([0,0,0]), nomChief = np.array([0,0,1])):
        
        rt = RayTrace.RayTrace(NraysCirc, NraysCross, rCirc, div_ang_x, div_ang_y, originChief, tiltChief, nomChief)
        
        self.Raytracer = rt
        
    def startRaytracer(self, surface):
        """
        Propagate rays in RayTracer to a surface.
        Adds new frame to rays, containing point of intersection and reflected direction.
        """
        
        self.system[surface].interpReflector()
        self.Raytracer.set_tcks(self.system[surface].tcks)
        self.Raytracer.propagateRays(a_init=100)
        
if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
