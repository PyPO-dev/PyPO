import math
import numpy as np

class RayTrace(object):
    def __init__(self, NraysCirc=0, NraysCross=0, 
                 rCirc=0, div_ang_x=np.pi, div_ang_y=np.pi,
                 originChief=np.array([0,0,0]), 
                 directionChief=np.array([0,0,1])):
        
        # This is the main ray container
        # ray_0 : chief ray
        # ray_1 - ray_(4*NraysCirc) : rays distributed in circle around chief ray (multiples of four)
        # ray_(4*NraysCirc+1) - ray_(4*NraysCirc + 4*NraysCross) : rays distributed in cross around chief ray (multiples of four)
        self.rays = {"ray_{}".format(n) : {} for n in range(1 + 4*NraysCirc + 4*NraysCross)}
        
        alpha = 0.0                             # Set first circle ray in the top of beam
        d_alpha = 2 * np.pi / (4 * NraysCirc)   # Set spacing in clockwise angle
        
        for i, (key, ray) in enumerate(self.rays.items()):
            #print(i)
            
            vec_in = np.array([0,0,1])
            
            ray["x_coords"] = [cold_focus[0],0,0,0, 0,0]
            ray["y_coords"] = [cold_focus[1],0,0,0, 0,0]
            ray["z_coords"] = [cold_focus[2],0,0,0, 0,0]
            
            # Add opening angles, not adding beamtilt yet!
            if i == 0:
                ray["angle0"] = [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
                ray["beam_focus"] = np.zeros(3) # Store this, dont change this
            
            elif i < self.N_rays:
                rotation = [theta_y * np.sin(alpha), theta_x * np.cos(alpha), 0]
                ray["angle0"] = [np.array(rotation), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
                ray["beam_focus"] = rotation # Store this, dont change this
                
                alpha += d_alpha
                
            else:
                if i < (self.N_rays + int(N_cross * 2)):
                    rotation = [0, beta_tot[i - self.N_rays], 0]
                    ray["angle0"] = [np.array(rotation), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
                    ray["beam_focus"] = rotation # Store this, dont change this
                
                else:
                    rotation = [beta_tot[i - self.N_rays], 0, 0]
                    ray["angle0"] = [np.array(rotation), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
                    ray["beam_focus"] = rotation # Store this, dont change this
        
        d_beta_x = div_ang_x / (NraysCross + 2) 
        d_beta_y = div_ang_y / (NraysCross + 2) 
        
    def __len__(self):
        return len(self.rays.keys())
    
    def __iter__(self):
        yield from self.rays.items()
        
    def __getitem__(self, key):
        return self.rays["ray_{}".format(key)]
    
    def __setitem__(self, key, ray):
        self.rays["ray_{}".format(key)] = ray
