import math
from functools import partial
import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp
import scipy.optimize as optimize

import src.Python.MatRotate as MatRotate

class RayTrace(object):
    def __init__(self, NraysCirc, NraysCross, 
                 rCirc, div_ang_x, div_ang_y,
                 originChief, tiltChief, nomChief):
        
        # This is the main ray container
        # ray_0 : chief ray
        # ray_1 - ray_(4*NraysCirc) : rays distributed in circle around chief ray (multiples of four)
        # ray_(4*NraysCirc+1) - ray_(4*NraysCirc + 4*NraysCross) : rays distributed in cross around chief ray (multiples of four)
        
        
        self.originChief = originChief
        self.directionChief = MatRotate.MatRotate(np.radians(tiltChief), nomChief, vecRot=True)
        
        self.rays = {"ray_{}".format(n) : {} for n in range(1 + 4*NraysCirc + 4*NraysCross)}
        
        if NraysCirc > 0:
            alpha = 0.0                             # Set first circle ray in the right of beam
            d_alpha = 2 * np.pi / (4 * NraysCirc)   # Set spacing in clockwise angle
        
        if NraysCross > 0:
            d_beta_x = div_ang_x / (NraysCross + 2) 
            d_beta_y = div_ang_y / (NraysCross + 2) 
            
            beta_x = np.linspace(d_beta_x, div_ang_x - d_beta_x, num=NraysCross)
            beta_y = np.linspace(d_beta_y, div_ang_y - d_beta_y, num=NraysCross)
            
            beta_x_pm = np.concatenate((-beta_x, beta_x))
            beta_y_pm = np.concatenate((-beta_y, beta_y))
            
            beta_tot = np.concatenate((beta_x_pm, beta_y_pm))
        
        # Ray initialization:
        # every ray gets two lists: one for position, one for direction
        # For each element encountered, a new entry will be made in each list:
        # we append a 3D array of the point to the position list and a 3D array to direction list
        for i, (key, ray) in enumerate(self.rays.items()):
            ray["positions"]  = []
            ray["directions"] = []
            if i == 0: # Chief ray
                ray["positions"].append(self.originChief)
                ray["directions"].append(self.directionChief)
                
            elif i > 0 and i <= 4 * NraysCirc:
                pos_ray = MatRotate.MatRotate(np.array([0,0,alpha]), np.array([rCirc,0,0]))
                
                rotation = np.array([np.radians(div_ang_x) * np.sin(alpha), np.radians(div_ang_y) * np.cos(alpha), 2*alpha])
                direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True)
                direction = MatRotate.MatRotate(np.radians(tiltChief), direction, vecRot=True)
    
                ray["positions"].append(self.originChief + pos_ray)
                ray["directions"].append(direction)
                
                alpha += d_alpha
                
            elif i > 4 * NraysCirc and i <= (4 * NraysCirc + 2 * NraysCross):
                beta_cur = beta_tot[i - 4 * NraysCirc - 1]
                pos_ray = np.array([0,rCirc,0])
                
                rotation = np.array([np.radians(div_ang_x) * np.sin(beta_cur), np.radians(div_ang_y) * np.cos(beta_cur), 0])
                direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True)
                ray["positions"].append(pos_ray)
        
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
        interp_u = interp.bisplev(x, y, self.tcks[1])
        interp_v = interp.bisplev(x, y, self.tcks[2])
        interp_w = interp.bisplev(x, y, self.tcks[3])
        return np.array([interp_u, interp_v, interp_w])
    
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
            
            a_opt = optimize.fmin(part_optLine_z, a_init, disp=0)
            #print(a_opt)
            
            position = a_opt*ray["directions"][-1] + ray["positions"][-1]
            #print(position[2])
            interp_n = self.interpEval_n(position[0], position[1])
            
            direction = ray["directions"][-1] - 2 * np.dot(ray["directions"][-1], interp_n) * interp_n
            #print(interp_n)
            ray["positions"].append(position)
            ray["directions"].append(direction)
    
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
