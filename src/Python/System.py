# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt

# POPPy-specific modules
import src.Python.Reflectors as Reflectors
import src.Python.RayTrace as RayTrace
import src.Python.Camera as Camera
import src.Python.Beams as Beams
import src.Python.PhysOptics as PO

class System(object):
    
    def __init__(self):
        self.num_ref = 0
        self.num_cam = 0
        self.system = {}
        
    def __str__(self):
        pass
    
    def __iter__(self):
        pass
    
    #### ADD REFLECTOR METHODS
    def addParabola(self, coef, mode='man', cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3), name="Parabola"):
        if name == "Parabola":
            name = name + "_{}".format(self.sys_id)
            
        if mode == 'man':
            a = coef[0]
            b = coef[1]
            
            p = Reflectors.Parabola(a, b, cRot, offTrans, offRot, name)
            
        elif mode == 'foc':
            f1 = coef[0] # Focal point position
            ve = coef[1] # Vertex position
            
            diff = f1 - ve

            df = np.sqrt(np.dot(diff, diff))
            a = 2 * np.sqrt(df)
            b = a
            
            orientation = diff / np.sqrt(np.dot(diff, diff))
            offTrans = ve
            
            # Find rotation in frame of vertex
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0#np.arccos(np.dot(np.array([0,0,1]), orientation))
        
            offRot = np.array([rx, ry, rz])
            cRot = offTrans
        
            p = Reflectors.Parabola(a, b, cRot, offTrans, offRot, name)
            
            p.focus_1 = f1
        
        self.system["{}".format(name)] = p
        self.num_ref += 1

    def addHyperbola(self, coef, mode='man', cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3), name="Hyperbola"):
        if name == "Hyperbola":
            name = name + "_{}".format(self.sys_id)
        
        if mode == 'man':
            a = coef[0]
            b = coef[1]
            c = coef[2]
            
            h = Reflectors.Hyperbola(a, b, c, cRot, offTrans, offRot, name)
        
        elif mode == 'foc':
            # Calculate a, b, c of hyperbola
            f1 = coef[0] # Focal point 1 position
            f2 = coef[1] # Focal point 2 position
            ecc = coef[2] # Eccentricity of hyperbola
            
            diff = f1 - f2
            c = np.sqrt(np.dot(diff, diff)) / 2
            a = c / ecc
            b = np.sqrt(c**2 - a**2)
            
            # Convert 2D hyperbola a,b,c to 3D hyperboloid a,b,c
            a3 = b
            b3 = b
            c3 = a
        
            # Find direction between focii
            orientation = diff / np.sqrt(np.dot(diff, diff))
        
            # Find offset from center. Use offset as rotation origin for simplicity
            center = (f1 + f2) / 2
            offTrans = center
        
            # Find rotation in frame of center
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0#np.arccos(np.dot(np.array([0,0,1]), orientation))
        
            offRot = np.array([rx, ry, rz])
            cRot = offTrans
        
            h = Reflectors.Hyperbola(a3, b3, c3, cRot, offTrans, offRot, name)
            
            h.focus_1 = f1
            h.focus_2 = f2
        
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def addEllipse_ab(self, name="Ellipse", a=2, b=3, c=5, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.sys_id)
        
        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot, name)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
        
    # Place and create ellipse using focus_1 and confocal distance c. Automatically creates a symmetric ellipsoid
    def addEllipse_foc(self, name="Ellipse", focus_1=np.array([0,0,1]), focus_2=np.array([0,0,1]), cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.sys_id)

        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot, name)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
    
    def addCamera(self, name="Camera", center=np.array([0,0,0]), offTrans=np.array([0,0,0]), offRot=np.array([0,0,0])):
        cam = Camera.Camera(center, offTrans, offRot, name)
        
        self.system["{}".format(name)] = cam
        self.num_cam += 1
        
    def removeElement(self, name):
        del self.system[name]
        self.num_ref -= 1
    
    #### PLOTTING OPTICAL SYSTEM
    def plotSystem(self, focus_1=False, focus_2=False, plotRaytrace=False):
        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
        
        for elem in self.system.values():
            if elem.elType == "Reflector":
                ax = elem.plotReflector(returns=True, ax_append=ax, focus_1=focus_1, focus_2=focus_2)
            
            elif elem.elType == "Camera":
                ax = elem.plotCamera(returns=True, ax_append=ax)
            
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
        
    def startRaytracer(self, surface, a_init=100, verbose=False):
        """
        Propagate rays in RayTracer to a surface.
        Adds new frame to rays, containing point of intersection and reflected direction.
        """
        if not hasattr(self.system[surface], 'tcks'):
            
            if self.system[surface].elType == "Reflector":
                self.system[surface].interpReflector(verbose=verbose)
                
            elif self.system[surface].elType == "Camera":
                self.system[surface].interpCamera()
        
        self.Raytracer.set_tcks(self.system[surface].tcks)
        self.Raytracer.propagateRays(a_init=a_init)
        
    def addBeam(self, x_lims, y_lims, gridsize, beam='pw', pol=np.array([1,0,0]), amp=1, phase=0, flip=False):
        if beam == 'pw':
            self.inputBeam = Beams.PlaneWave(x_lims, y_lims, gridsize, pol, amp, phase, flip)
            
    #### TODO: make following two functions look less horrible
    def initPhysOptics(self, target, k, thres=-50, numThreads=1, inputPath='./src/C++/input/', outputPath='./src/C++/output/'):
        """
        Create a PO object that will act as the 'translator' between POPPy and PhysBeam.
        Also performs the initial propagation from beam to target.
        Should be called once per system.
        """
        
        self.PO = PO.PhysOptics(inputPath, outputPath, k, numThreads, thres)

        # Write beam to input
        self.PO.writeInput('grid_s_x.txt', self.inputBeam.grid_x)
        self.PO.writeInput('grid_s_y.txt', self.inputBeam.grid_y)
        self.PO.writeInput('grid_s_z.txt', self.inputBeam.grid_z)
        
        self.PO.writeInput('rJs_x.txt', np.real(self.inputBeam.Jx))
        self.PO.writeInput('rJs_y.txt', np.real(self.inputBeam.Jy))
        self.PO.writeInput('rJs_z.txt', np.real(self.inputBeam.Jz))
        
        self.PO.writeInput('iJs_x.txt', np.imag(self.inputBeam.Jx))
        self.PO.writeInput('iJs_y.txt', np.imag(self.inputBeam.Jy))
        self.PO.writeInput('iJs_z.txt', np.imag(self.inputBeam.Jz))
        
        self.PO.writeInput('rMs_x.txt', np.real(self.inputBeam.Mx))
        self.PO.writeInput('rMs_y.txt', np.real(self.inputBeam.My))
        self.PO.writeInput('rMs_z.txt', np.real(self.inputBeam.Mz))
        
        self.PO.writeInput('iMs_x.txt', np.imag(self.inputBeam.Mx))
        self.PO.writeInput('iMs_y.txt', np.imag(self.inputBeam.My))
        self.PO.writeInput('iMs_z.txt', np.imag(self.inputBeam.Mz))
        
        # Write target grid
        self.PO.writeInput('grid_t_x.txt', target.grid_x)
        self.PO.writeInput('grid_t_y.txt', target.grid_y)
        self.PO.writeInput('grid_t_z.txt', target.grid_z)
        
        self.PO.writeInput('norm_t_nx.txt', target.grid_nx)
        self.PO.writeInput('norm_t_ny.txt', target.grid_ny)
        self.PO.writeInput('norm_t_nz.txt', target.grid_nz)
        
        self.PO.writeInput('As.txt', target.area)

    def nextPhysOptics(self, target):
        """
        Perform a physical optics propagation from source to target.
        Automatically continues from last propagation by copying currents and gird to input
        """
        self.PO.copyToInput('grid_t_x.txt', 'grid_s_x.txt', grid=True)
        self.PO.copyToInput('grid_t_y.txt', 'grid_s_y.txt', grid=True)
        self.PO.copyToInput('grid_t_z.txt', 'grid_s_z.txt', grid=True)
        
        self.PO.copyToInput('rJt_x.txt', 'rJs_x.txt')
        self.PO.copyToInput('rJt_y.txt', 'rJs_y.txt')
        self.PO.copyToInput('rJt_z.txt', 'rJs_z.txt')
        
        self.PO.copyToInput('iJt_x.txt', 'iJs_x.txt')
        self.PO.copyToInput('iJt_y.txt', 'iJs_y.txt')
        self.PO.copyToInput('iJt_z.txt', 'iJs_z.txt')
        
        self.PO.copyToInput('rMt_x.txt', 'rMs_x.txt')
        self.PO.copyToInput('rMt_y.txt', 'rMs_y.txt')
        self.PO.copyToInput('rMt_z.txt', 'rMs_z.txt')
        
        self.PO.copyToInput('iMt_x.txt', 'iMs_x.txt')
        self.PO.copyToInput('iMt_y.txt', 'iMs_y.txt')
        self.PO.copyToInput('iMt_z.txt', 'iMs_z.txt')
        
        # Write target grid
        self.PO.writeInput('grid_t_x.txt', target.grid_x)
        self.PO.writeInput('grid_t_y.txt', target.grid_y)
        self.PO.writeInput('grid_t_z.txt', target.grid_z)
        
        self.PO.writeInput('norm_t_nx.txt', target.grid_nx)
        self.PO.writeInput('norm_t_ny.txt', target.grid_ny)
        self.PO.writeInput('norm_t_nz.txt', target.grid_nz)
        
        self.PO.writeInput('As.txt', target.area)
    
    def runPhysOptics(self, save=0):
        self.PO.runPhysOptics(save)
        
if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
