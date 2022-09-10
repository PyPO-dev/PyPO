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
                 originChief, tiltChief, nomChief, mode):
        
        # This is the main ray container
        # ray_0 : chief ray
        # ray_1 - ray_(4*NraysCirc) : rays distributed in circle around chief ray (multiples of four)
        # ray_(4*NraysCirc+1) - ray_(4*NraysCirc + 4*NraysCross) : rays distributed in cross around chief ray (multiples of four)
        
        
        self.originChief = originChief
        self.directionChief = MatRotate.MatRotate(np.radians(tiltChief), nomChief, vecRot=True)
        
        self.nTot = 1 + nCirc * 4 * nRays
        
        self.rays = {"ray_{}".format(n) : {} for n in range(self.nTot)}

        alpha = 0.0                             # Set first circle ray in the right of beam
        if nRays > 0:
            
            d_alpha = 2 * np.pi / (4 * nRays)   # Set spacing in clockwise angle
            
        else:
            d_alpha = 0
        
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
                
                if mode == 'auto':
                    mode = self.getMode()
                
                #if nRays == 0:
                continue
            
            if mode == 'z':
                pos_ray = MatRotate.MatRotate(np.array([0,0,alpha]), np.array([rCirc / nCirc * n,0,0]), radians=True)
                rotation = np.array([np.radians(div_ang_x) * np.sin(alpha), np.radians(div_ang_y) * np.cos(alpha), 2*alpha]) / nCirc * n
                
            elif mode == 'x':
                pos_ray = MatRotate.MatRotate(np.array([alpha,0,0]), np.array([0,rCirc / nCirc * n,0]), radians=True)
                rotation = np.array([2*alpha, np.radians(div_ang_x) * np.sin(alpha), np.radians(div_ang_y) * np.cos(alpha)]) / nCirc * n
                
            elif mode == 'y':
                pos_ray = MatRotate.MatRotate(np.array([0,alpha,0]), np.array([0,0,rCirc / nCirc * n]), radians=True)
                rotation = np.array([np.radians(div_ang_y) * np.cos(alpha), np.radians(div_ang_x) * np.sin(alpha), 2*alpha]) / nCirc * n
            
            direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True, radians=True)
            direction = MatRotate.MatRotate(tiltChief, direction, vecRot=True)
    
            ray["positions"].append(self.originChief + pos_ray)
            ray["directions"].append(direction)

            alpha += d_alpha
            
            if i == int(self.nTot / nCirc) * n:
                #print(int(self.nTot / nCirc) * n)
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
        
    def getMode(self):
        """
        Obtain axis for ray trace interpolation. 
        Takes dominant components of chief ray as propagation axis.
        
        @return mode The dominant ray axis used for interpolation
        """
        
        axes = ['x', 'y', 'z']
        axis = np.argmax(np.absolute(self.rays["ray_0"]["directions"][-1]))
        mode = axes[axis]
        return mode
        
    def interpEval(self, x1, x2):
        x3 = interp.bisplev(x1, x2, self.tcks[0])
        return x3
        
    def interpEval_n(self, x1, x2):
        interp_nx = interp.bisplev(x1, x2, self.tcks[1])
        interp_ny = interp.bisplev(x1, x2, self.tcks[2])
        interp_nz = interp.bisplev(x1, x2, self.tcks[3])
        
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
        
        interp_z = self.interpEval(position[0], position[1])
        
        return np.absolute(interp_z - position[2])
    
    def optLine_x(self, a, ray):
        """
        This function calculates the difference between a line x-coordinate and the 
        interpolated mirror x-coordinate. Should be used inside optimize.fmin functions
        """
        
        position = a * ray["directions"][-1] + ray["positions"][-1]
        
        interp_x = self.interpEval(position[1], position[2])
        
        return np.absolute(interp_x - position[0])
    
    def optLine_y(self, a, ray):
        """
        This function calculates the difference between a line x-coordinate and the 
        interpolated mirror y-coordinate. Should be used inside optimize.fmin functions
        """
        
        position = a * ray["directions"][-1] + ray["positions"][-1]
        
        interp_y = self.interpEval(position[2], position[0])
        
        return np.absolute(interp_y - position[1])
    
    def propagateRays(self, a_init, mode):
        for i, (key, ray) in enumerate(self.rays.items()):

            if mode == 'z':
                part_optLine = partial(self.optLine_z, ray=ray)
                x1 = 0
                x2 = 1
                
            elif mode == 'x':
                part_optLine = partial(self.optLine_x, ray=ray)
                x1 = 1
                x2 = 2
                
            elif mode == 'y':
                part_optLine = partial(self.optLine_y, ray=ray)
                x1 = 2
                x2 = 0
                
            a_opt = optimize.fmin(part_optLine, a_init, disp=0, xtol=1e-10, ftol=1e-10)
            
            position = a_opt*ray["directions"][-1] + ray["positions"][-1]
            
            interp_n = self.interpEval_n(position[x1], position[x2])
            
            direction = ray["directions"][-1] - 2 * np.dot(ray["directions"][-1], interp_n) * interp_n
            
            norm = np.sqrt(np.dot(direction, direction))
            
            ray["positions"].append(position)
            ray["directions"].append(direction)
    
    def plotRays(self, quiv=True, frame=0, mode='z'):
        fig, ax = pt.subplots(1,1)
        
        for i, (key, ray) in enumerate(self.rays.items()):
            '''
            if i == 1:
                ax.scatter(ray["positions"][-1][0], ray["positions"][-1][1], color='red', s=1)
                
            else:
                ax.scatter(ray["positions"][-1][0], ray["positions"][-1][1], color='black')
            '''
            if mode == 'z':
                ax.scatter(ray["positions"][frame][0], ray["positions"][frame][1], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][0], ray["positions"][frame][1], 10 * ray["directions"][frame][0], 10 * ray["directions"][frame][1], color='black', width=0.005, scale=1000)
                    
            elif mode == 'x':
                ax.scatter(ray["positions"][frame][1], ray["positions"][frame][2], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][1], ray["positions"][frame][2], 10 * ray["directions"][frame][1], 10 * ray["directions"][frame][2], color='black', width=0.005, scale=1000)
                    
            elif mode == 'y':
                ax.scatter(ray["positions"][frame][2], ray["positions"][frame][0], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][2], ray["positions"][frame][0], 10 * ray["directions"][frame][2], 10 * ray["directions"][frame][0], color='black', width=0.005, scale=1000)
            
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
