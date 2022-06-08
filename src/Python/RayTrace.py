import math
from functools import partial
import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp
import scipy.optimize as optimize

import src.Python.MatRotate as MatRotate

class RayTrace(object):
    def __init__(self, nRays, nCirc, 
                 rCirc, div_ang_x, div_ang_y,
                 originChief, tiltChief, nomChief):
        
        # This is the main ray container
        # ray_0 : chief ray
        # ray_1 - ray_(4*NraysCirc) : rays distributed in circle around chief ray (multiples of four)
        # ray_(4*NraysCirc+1) - ray_(4*NraysCirc + 4*NraysCross) : rays distributed in cross around chief ray (multiples of four)
        
        
        self.originChief = originChief
        self.directionChief = MatRotate.MatRotate(np.radians(tiltChief), nomChief, vecRot=True)
        
        self.nTot = 1 + nCirc * 4 * nRays
        
        self.rays = {"ray_{}".format(n) : {} for n in range(self.nTot)}

        
        if nRays > 0:
            alpha = 0.0                             # Set first circle ray in the right of beam
            d_alpha = 2 * np.pi / (4 * nRays)   # Set spacing in clockwise angle
        
        # Ray initialization:
        # every ray gets two lists: one for position, one for direction
        # For each element encountered, a new entry will be made in each list:
        # we append a 3D array of the point to the position list and a 3D array to direction list
        
        n = 1
        
        for i, (key, ray) in enumerate(self.rays.items()):
            ray["positions"]  = []
            ray["directions"] = []
            
            if i == 0: # Chief ray
                ray["positions"].append(self.originChief)
                ray["directions"].append(self.directionChief)

            pos_ray = MatRotate.MatRotate(np.array([0,0,alpha]), np.array([rCirc / nCirc * n,0,0]))
                
            rotation = np.array([np.radians(div_ang_x) * np.sin(alpha), np.radians(div_ang_y) * np.cos(alpha), 2*alpha]) / nCirc * n
            direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True)
            direction = MatRotate.MatRotate(np.radians(tiltChief), direction, vecRot=True)
    
            ray["positions"].append(self.originChief + pos_ray)
            ray["directions"].append(direction)
                
            alpha += d_alpha
            
            if i == int(self.nTot / nCirc) * n:
                print(int(self.nTot / nCirc) * n)
                n += 1
                alpha = 0
        
    def __len__(self):
        return len(self.rays.keys())
    
    def __iter__(self):
        yield from self.rays.items()
        
    def __getitem__(self, key):
        return self.rays["ray_{}".format(key)]
    
    def __setitem__(self, key, ray):
        self.rays["ray_{}".format(key)] = ray
        
    def __str__(self):
        s = """\n######################### RAYTRACER INFO #########################
Chief ray origin    : [{:.3f}, {:.3f}, {:.3f}] [mm]
Chief ray direction : [{:.4f}, {:.4f}, {:.4f}] 
######################### RAYTRACER INFO #########################\n""".format(self.originChief[0], self.originChief[1], self.originChief[2],
                                                                               self.directionChief[0], self.directionChief[1], self.directionChief[2])
        return s
    
    def set_tcks(self, tcks):
        self.tcks = tcks
        
    def interpEval_z(self, x, y):
        interp_z = interp.bisplev(x, y, self.tcks[0])
        return interp_z
        
    def interpEval_n(self, x, y):
        interp_nx = interp.bisplev(x, y, self.tcks[1])
        interp_ny = interp.bisplev(x, y, self.tcks[2])
        interp_nz = interp.bisplev(x, y, self.tcks[3])
        
        norm = np.sqrt(interp_nz**2 + interp_ny**2 + interp_nx**2)
        
        interp_nx /= norm
        interp_ny /= norm
        interp_nz /= norm
        
        return np.array([interp_nx, interp_ny, interp_nz])
    
    def optLine_z(self, a, ray):
        """
        This function calculates the difference between a line z-coordinate and the 
        interpolated mirror z-coordinate. Should be used inside optimize.fmin functions
        """
        
        position = a * ray["directions"][-1] + ray["positions"][-1]
        
        interp_z = self.interpEval_z(position[0], position[1])
        
        return np.absolute(interp_z - position[2])
    
    def propagateRays(self, a_init):
        for i, (key, ray) in enumerate(self.rays.items()):
            part_optLine_z = partial(self.optLine_z, ray=ray)
            
            a_opt = optimize.fmin(part_optLine_z, a_init, disp=0, xtol=1e-10, ftol=1e-10)
            
            position = a_opt*ray["directions"][-1] + ray["positions"][-1]

            interp_n = self.interpEval_n(position[0], position[1])
            
            direction = ray["directions"][-1] - 2 * np.dot(ray["directions"][-1], interp_n) * interp_n
            
            norm = np.sqrt(np.dot(direction, direction))
            #direction /= norm
            '''
            # Diagnostics - remove when fixed
            print("Incoming ray direction at surface:")
            print(ray["directions"][-1])
            print("Normal vector at surface:")
            print(interp_n)
            print("ray direction at surface")
            print(direction)
            
            '''
            
            
            ray["positions"].append(position)
            ray["directions"].append(direction)
            
        print(self.rays["ray_4"]["positions"][-1])
    
    def plotRays(self, quiv=True, frame=0):
        fig, ax = pt.subplots(1,1)
        
        for i, (key, ray) in enumerate(self.rays.items()):
            '''
            if i == 1:
                ax.scatter(ray["positions"][-1][0], ray["positions"][-1][1], color='red', s=1)
                
            else:
                ax.scatter(ray["positions"][-1][0], ray["positions"][-1][1], color='black')
            '''
            ax.scatter(ray["positions"][frame][0], ray["positions"][frame][1], color='black', s=10)
            
            if quiv:
                ax.quiver(ray["positions"][frame][0], ray["positions"][frame][1], 10 * ray["directions"][frame][0], 10 * ray["directions"][frame][1], color='black', width=0.005, scale=10)
            
        ax.set_aspect(1)
        pt.show()
        
    def plotRaysSystem(self, ax_append):
        
        for i, (key, ray) in enumerate(self.rays.items()):
            x = []
            y = []
            z = []
            for j in range(len(ray["positions"])):
                
                x.append(ray["positions"][j][0])
                y.append(ray["positions"][j][1])
                z.append(ray["positions"][j][2])
                
            ax_append.plot(x, y, z, color='grey')
            
        return ax_append

if __name__ == "__main__":
    tiltChief=np.array([0,1,-2.5])
    tiltChief=np.array([0,0,0])
    
    rt = RayTrace(NraysCirc=10, rCirc=1, tiltChief=tiltChief, NraysCross=0)
    print(rt)
    rt.plotInitialBeam()
