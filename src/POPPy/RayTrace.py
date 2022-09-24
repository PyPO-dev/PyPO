import math
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp
import scipy.optimize as optimize
from itertools import islice
from multiprocessing import  Pool
from functools import partial

import src.POPPy.MatRotate as MatRotate

class RayTrace(object):
    def __init__(self):
        pass
                
    def initRaytracer(self, nRays, nRing, 
                 a, b, angx, angy,
                 originChief, tiltChief):
        
        # This is the main ray container
        # ray_0 : chief ray
        # ray_1 - ray_(4*NraysCirc) : rays distributed in circle around chief ray (multiples of four)
        # ray_(4*NraysCirc+1) - ray_(4*NraysCirc + 4*NraysCross) : rays distributed in cross around chief ray (multiples of four)
        
        nomChief = np.array([0,0,1]) # Always initialize raytrace beam along z-axis
        
        self.originChief = originChief
        self.directionChief = MatRotate.MatRotate(tiltChief, nomChief, vecRot=True)
        
        self.nTot = 1 + nRing * 4 * nRays
        
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

                continue
            
            pos_ray = np.array([a * np.cos(alpha), b * np.sin(alpha), 0]) / nRing * n
            rotation = np.array([np.radians(angy) * np.sin(alpha) / nRing * n, np.radians(angx) * np.cos(alpha) / nRing * n, 2*alpha])

            direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True, radians=True)
            direction = MatRotate.MatRotate(tiltChief, direction, vecRot=True)
    
            pos_r = MatRotate.MatRotate(tiltChief, self.originChief + pos_ray, origin=self.originChief)
    
            ray["positions"].append(pos_r)
            ray["directions"].append(direction)

            alpha += d_alpha
            
            if i == int(self.nTot / nRing) * n:
                n += 1
                alpha = 0
        
    def POtoRaytracer(self, source, Pr):
        x = source.grid_x.flatten()
        y = source.grid_y.flatten()
        z = source.grid_z.flatten()
        
        px = Pr[0].flatten()
        py = Pr[1].flatten()
        pz = Pr[2].flatten()
        
        self.nTot = len(x)
        
        self.rays = {"ray_{}".format(n) : {} for n in range(self.nTot)}
        
        for i, (key, ray) in enumerate(self.rays.items()):
            ray["positions"]  = []
            ray["directions"] = []
            
            pos = np.array([x[i], y[i], z[i]])
            direction = np.array([px[i], py[i], pz[i]])
            
            ray["positions"].append(pos)
            ray["directions"].append(direction)
        
    def __len__(self):
        return len(self.rays.keys())
    
    def __iter__(self):
        yield from self.rays.items()
        
    def __getitem__(self, key):
        return self.rays["ray_{}".format(key)]
    
    def __setitem__(self, key, ray):
        self.rays["ray_{}".format(key)] = ray
        
    def sizeOf(self, units='b'):
        """
        Return size of self.rays.
        """
        
        factor = 1
        
        if units == 'kb':
            factor = 1e-3
        
        elif units == 'mb':
            factor = 1e-6
            
        elif units == 'gb':
            factor = 1e-9
            
        size = sys.getsizeof(self.rays)
        
        for i, (key, ray) in enumerate(self.rays.items()):
            # Add size of reference to ray
            size += sys.getsizeof(ray)
            
            for key, attr in ray.items():
                # Add size of reference to ray
                size += sys.getsizeof(attr)
                
                for coord in attr:
                    size += sys.getsizeof(coord)

        return size * factor
        
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
    
    def optLine(self, a, ray, x1, x2, x3):
        """
        This function calculates the difference between a line xyz-coordinate and the 
        interpolated mirror xyz-coordinate. Should be used inside optimize.fmin functions
        """
        
        position = a * ray["directions"][-1] + ray["positions"][-1]
        
        interp_z = self.interpEval(position[x1], position[x2])
        
        return np.absolute(interp_z - position[x3])
    
    def chunkify(self, data, s):
        it = iter(data)
        for i in range(0, len(data), s):
            yield {k:data[k] for k in islice(it, s)}
        
    
    def propagateRays(self, a0, mode, workers):
        
        if mode == 'z':
            x1 = 0
            x2 = 1
            x3 = 2
                
        elif mode == 'x':
            x1 = 1
            x2 = 2
            x3 = 0
                
        elif mode == 'y':
            x1 = 2
            x2 = 0
            x3 = 1
        
        self.nReduced = int(self.nTot / workers)
        
        # Evaluate _propagateRays partially
        _propagateRayspar = partial(self._propagateRays, a0=a0,
                                    x1=x1, x2=x2, x3=x3)
        
        PIDs = np.arange(workers)

        # Pack rays into subdicts
        subrays_l = []

        for subrays in self.chunkify(self.rays, self.nReduced):
            subrays_l.append(subrays)
            
        args = zip(subrays_l, PIDs)
        pool = Pool(workers)

        toStore = pool.map(_propagateRayspar, args)
        pool.close()
        
        for _toStore in toStore:
            self.rays.update(_toStore)

    def _propagateRays(self, args, a0, x1, x2, x3):
        rays, PID = args
        
        toStore = rays
        
        j = 0 # Counter index
            
        for i, (key, ray) in enumerate(rays.items()):
            part_optLine = partial(self.optLine, ray=ray, x1=x1, x2=x2, x3=x3)
                
            a_opt = optimize.fmin(part_optLine, a0, disp=0, xtol=1e-16, ftol=1e-16)
            
            position = a_opt*ray["directions"][-1] + ray["positions"][-1]
            
            interp_n = self.interpEval_n(position[x1], position[x2])
            
            direction = ray["directions"][-1] - 2 * np.dot(ray["directions"][-1], interp_n) * interp_n

            toStore[key]["positions"].append(position)
            toStore[key]["directions"].append(direction)

            if (i/self.nReduced * 100) > j and PID == 0:
                print("{} / 100".format(j), end='\r')
                j += 1
                
        return toStore
    
    def calcPathlength(self):
        """
        Adds total path length as dict entry for each ray
        """
        for i, (key, ray) in enumerate(self.rays.items()):
            stride = np.zeros(3)
            L = 0
            
            for i, pos in enumerate(ray["positions"]):
                if i == 0:
                    stride = pos
                    continue
                
                else:
                    diff = pos - stride
                    L += np.sqrt(np.dot(diff, diff))
                    stride = pos
                    
            ray["length"] = L
        
    def plotRays(self, quiv=True, frame=0, mode='z', save=False):
        fig, ax = pt.subplots(1,1)
        
        for i, (key, ray) in enumerate(self.rays.items()):

            if mode == 'z':
                ax.set_xlabel('$x$ / [mm]')
                ax.set_ylabel('$y$ / [mm]')
                ax.scatter(ray["positions"][frame][0], ray["positions"][frame][1], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][0], ray["positions"][frame][1], 10 * ray["directions"][frame][0], 10 * ray["directions"][frame][1], color='black', width=0.005, scale=10)
                    
            elif mode == 'x':
                ax.set_xlabel('$y$ / [mm]')
                ax.set_ylabel('$z$ / [mm]')
                ax.scatter(ray["positions"][frame][1], ray["positions"][frame][2], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][1], ray["positions"][frame][2], 10 * ray["directions"][frame][1], 10 * ray["directions"][frame][2], color='black', width=0.005, scale=10)
                    
            elif mode == 'y':
                ax.set_xlabel('$z$ / [mm]')
                ax.set_ylabel('$x$ / [mm]')
                ax.scatter(ray["positions"][frame][2], ray["positions"][frame][0], color='black', s=10)
            
                if quiv:
                    ax.quiver(ray["positions"][frame][2], ray["positions"][frame][0], 10 * ray["directions"][frame][2], 10 * ray["directions"][frame][0], color='black', width=0.005, scale=10)
                    
        ax.set_aspect(1)
        ax.set_title('Spot diagram, frame = {}'.format(frame))
        
        if save:
            pt.savefig(fname='spot_{}.jpg'.format(frame),bbox_inches='tight', dpi=300)

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
