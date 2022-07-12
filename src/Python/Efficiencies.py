import numpy as np

class Efficiencies(object):
    
    def calcSpillover(self, surfaceObject, field, aperture):
        x = surfaceObject.grid_x
        y = surfaceObject.grid_y
        
        field_ap = field * aperture.astype(complex)
        
        eff_s = np.absolute(np.sum(np.conj(field_ap) * field)**2) / (np.sum(np.absolute(field)**2) * np.sum(np.absolute(field_ap)**2))
        
        return eff_s
    
    def calcTaper(self, surfaceObject, field, aperture):
        x = surfaceObject.grid_x
        y = surfaceObject.grid_y
        area = surfaceObject.area
        
        print(field[aperture].shape)
        
        field_ap = field * aperture.astype(complex)
        
        eff_t = np.absolute(np.sum(field[aperture] * area[aperture]))**2 / np.sum(np.absolute(field[aperture])**2 * area[aperture]) / np.sum(aperture*area)
        
        return eff_t
