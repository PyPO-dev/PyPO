import numpy as np
import scipy.optimize as opt

class Fitting(object):
    
    def __init__():
        pass
    
    def fitGaussAbs(self, field, surfaceObject, plane='xy', thres):
        if plane == 'xy':
            x = surfaceObject.grid_x
            y = surfaceObject.grid_y
            
        x0 = np.exp(-4)
        y0 = np.exp(-4)
        
        pars = np.array([x0, y0])
        args = zip(field, x, y)

        out = opt.fmin(self.couplingAbs, pars, args)
        
        return out
        
    def couplingAbs(self, pars, *args):
        x0, y0 = pars
        field, x, y = args
        field = np.absolute(field)
        
        # Normalize field
        field_norm = field / np.max(field)
        
        Psi = GaussAbs(x, y, x0, y0)
        
        coupling = np.sum(np.absolute(Psi)**2) / np.sum(np.absolute(field_norm)**2)
        
        return 1 - coupling
        
        
        
            
    def GaussAbs(self, x, y, x0, y0):
        Psi = np.exp(-(x / x0)**2 -(y/y0)**2)
        return Psi
        
