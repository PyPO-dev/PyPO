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
import src.Python.FourierOptics as FO
from src.Python.Plotter import Plotter
import src.Python.Aperture as Aperture
#from src.Python.Fitting import Fitting

class System(object):
    
    fileNames_s = ['grid_s_x.txt', 'grid_s_y.txt', 'grid_s_z.txt', 'As.txt', 'Js_x.txt', 'Js_y.txt', 'Js_z.txt', 'Ms_x.txt', 'Ms_y.txt', 'Ms_z.txt']
    
    fileNames_t = ['grid_t_x.txt', 'grid_t_y.txt', 'grid_t_z.txt', 'As.txt', 'norm_t_nx.txt', 'norm_t_ny.txt', 'norm_t_nz.txt']
    
    fileNames_tc = ['Jt_x.txt', 'Jt_y.txt', 'Jt_z.txt', 'Mt_x.txt', 'Mt_y.txt', 'Mt_z.txt']
    
    customBeamPath = './custom/beam/'
    customReflPath = './custom/reflector/'
    
    def __init__(self):
        self.num_ref = 0
        self.num_cam = 0
        self.system = {}
        
    def __str__(self):
        pass
    
    def setCustomBeamPath(self, path, append=False):
        if append:
            self.customBeamPath += path
        else:
            self.customBeamPath = path
            
    def setCustomReflPath(self, path, append=False):
        if append:
            self.customReflPath += path
        else:
            self.customReflPath = path
            
    def addPlotter(self, save='./images/'):
        self.plotter = Plotter(save='./images/')
    
    #### ADD REFLECTOR METHODS
    def addParabola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Parabola", axis='a', trunc=False, flip=False):
        """
        Function for adding paraboloid reflector to system. If gmode='uv', lims_x should contain the smallest and largest radius and lims_y 
        should contain rotation.
        """
        if name == "Parabola":
            name = name + "_{}".format(self.num_ref)
            
        if pmode == 'man':
            a = coef[0]
            b = coef[1]
            
            p = Reflectors.Parabola(a, b, cRot, name)
            
            p.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
        elif pmode == 'foc':
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
        
            p = Reflectors.Parabola(a, b, cRot, name)
            
            p.focus_1 = f1
            p.focus_2 = np.ones(3) * float("NaN")
            
            p.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
            p.rotateGrid(offRot)
            p.translateGrid(offTrans)
        
        self.system["{}".format(name)] = p
        self.num_ref += 1

    def addHyperbola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Hyperbola", axis='a', trunc=False, flip=False):
        if name == "Hyperbola":
            name = name + "_{}".format(self.num_ref)
        
        if pmode == 'man':
            a = coef[0]
            b = coef[1]
            c = coef[2]
            
            h = Reflectors.Hyperbola(a, b, c, cRot, name)

            h.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
        
        elif pmode == 'foc':
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
        
            h = Reflectors.Hyperbola(a3, b3, c3, cRot, name)
            
            h.setGrid(lims_x, lims_y, gridsize, gmode, trunc, flip, axis)
            
            h.rotateGrid(offRot)
            h.translateGrid(offTrans)
            
            h.focus_1 = f1
            h.focus_2 = f2
            
        self.system["{}".format(name)] = h
        self.num_ref += 1
        
    def addEllipse_ab(self, name="Ellipse", a=2, b=3, c=5, cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.num_ref)
        
        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot, name)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
        
    # Place and create ellipse using focus_1 and confocal distance c. Automatically creates a symmetric ellipsoid
    def addEllipse_foc(self, name="Ellipse", focus_1=np.array([0,0,1]), focus_2=np.array([0,0,1]), cRot=np.zeros(3), offTrans=np.zeros(3), offRot=np.zeros(3)):
        if name == "Ellipse":
            name = name + "_{}".format(self.num_ref)

        e = Reflectors.Ellipse(a, b, c, cRot, offTrans, offRot, name)
        
        self.system["{}".format(name)] = e
        self.num_ref += 1
    
    def addCustomReflector(self, location, name="Custom", cRot=np.zeros(3), offTrans=np.zeros(3)):
        if name == "Custom":
            name = name + "_{}".format(self.num_ref)
            
        path = self.customReflPath + location
        custom = Reflectors.Custom(cRot, name, path)
        
        self.system["{}".format(name)] = custom
        self.num_ref += 1
    
    def addCamera(self, lims_x, lims_y, gridsize, center=np.zeros(3), name="Camera"):
        cam = Camera.Camera(center, name)
        
        cam.setGrid(lims_x, lims_y, gridsize)
        
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
                
            elif elem.elType == "Aperture":
                ax = elem.plotAperture(returns=True, ax_append=ax)
            
        if plotRaytrace:
            ax = self.Raytracer.plotRaysSystem(ax_append=ax)
            
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        pt.show()
        
    def initRaytracer(self, nRays=0, nCirc=1, 
                 rCirc=0, div_ang_x=2*np.pi, div_ang_y=2*np.pi,
                 originChief=np.array([0,0,0]), 
                 tiltChief=np.array([0,0,0]), nomChief = np.array([0,0,1])):
        
        rt = RayTrace.RayTrace(nRays, nCirc, rCirc, div_ang_x, div_ang_y, originChief, tiltChief, nomChief)
        
        self.Raytracer = rt
        
    def startRaytracer(self, surface, a_init=100):
        """
        Propagate rays in RayTracer to a surface.
        Adds new frame to rays, containing point of intersection and reflected direction.
        """
        if not hasattr(self.system[surface], 'tcks'):
            
            if self.system[surface].elType == "Reflector":
                self.system[surface].interpReflector()
                
            elif self.system[surface].elType == "Camera":
                self.system[surface].interpCamera()
        
        self.Raytracer.set_tcks(self.system[surface].tcks)
        self.Raytracer.propagateRays(a_init=a_init)
        
    def addBeam(self, lims_x, lims_y, gridsize, beam='pw', pol=np.array([1,0,0]), amp=1, phase=0, flip=False, name='', comp='Ex'):
        if beam == 'pw':
            self.inputBeam = Beams.PlaneWave(lims_x, lims_y, gridsize, pol, amp, phase, flip, name)
            
        elif beam == 'custom':
            pathsToFields = [self.customBeamPath + 'r' + name, self.customBeamPath + 'i' + name, self.customBeamPath]
            self.inputBeam = Beams.CustomBeam(lims_x, lims_y, gridsize, comp, pathsToFields, flip, name)
            
    def addCustomBeamGrid(, comp, pathsToField, flip, name)
            
    def addPointSource(self, area=1, pol=np.array([1,0,0]), amp=1, phase=0, flip=False, name='', n=3):
        self.inputBeam = Beams.PointSource(area, pol, amp, phase, flip, name, n)
            
    def initPhysOptics(self, target, k, thres=-50, numThreads=1, cpp_path='./src/C++/', contd=False):
        """
        Create a PO object that will act as the 'translator' between POPPy and PhysBeam.
        Also performs the initial propagation from beam to target.
        Should be called once per system.
        """
        
        self.PO = PO.PhysOptics(k, numThreads, thres, cpp_path)

        # Write input beam to input
        for i, attr in enumerate(self.inputBeam):
            self.PO.writeInput(self.fileNames_s[i], attr)
            
        for i, attr in enumerate(target):
            if i == 3:
                # Dont write area of target
                pass
            else:
                self.PO.writeInput(self.fileNames_t[i], attr)

    def nextPhysOptics(self, source, target):
        """
        Perform a physical optics propagation from source to target.
        Automatically continues from last propagation by copying currents and gird to input
        """
        for i, attr in enumerate(source):
            # Only write xyz and area
            if i <= 3:
                self.PO.writeInput(self.fileNames_s[i], attr)
            else:
                continue
        
        for i, field in enumerate(self.fileNames_tc):
            # Copy J and M from output to input
            self.PO.copyToInput(self.fileNames_tc[i], self.fileNames_s[i+4])
        
        # Write target grid
        for i, attr in enumerate(target):
            if i == 3:
                # Dont write area of target
                pass
            else:
                self.PO.writeInput(self.fileNames_t[i], attr)
    
    def runPhysOptics(self, save=0, material_source='vac', prop_mode=0):
        self.PO.runPhysOptics(save, material_source, prop_mode)
        
    def loadField(self, surface, mode='Ex'):
        field = self.PO.loadField(surface.shape, mode)
        
        return field
        
    def initFourierOptics(self, k):
        self.FO = FO.FourierOptics(k=k)
        
    def addCircAper(self, r_max, gridsize, r_min=1e-3, cRot=np.zeros(3), name=''):
        ap = Aperture.Aperture(cRot, name)
        ap.makeCircAper(r_max, r_min, gridsize)
        self.system["{}".format(name)] = ap
        
    
    '''
    def addFitter(self):
        self.FI = Fitting()
        
    def fitGauss(self, mode='abs', thres=-11):
        #TODO Implement fitting object
        pass
    '''    
if __name__ == "__main__":
    print("Please run System.py from the SystemInterface.py, located in the POPPy directory.")
