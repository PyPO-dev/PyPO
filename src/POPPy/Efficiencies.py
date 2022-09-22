import numpy as np

class Efficiencies(object):
    
    def calcSpillover(self, surfaceObject, field, R_aper, ret_field):
        field = field[0]
        x = surfaceObject.grid_x
        y = surfaceObject.grid_y
        
        maskAper = np.sqrt(x**2 + y**2) < R_aper

        field_ap = field * maskAper.astype(complex)
        
        eff_s = np.absolute(np.sum(np.conj(field_ap) * field)**2) / (np.sum(np.absolute(field)**2) * np.sum(np.absolute(field_ap)**2))
        
        if ret_field:
            return eff_s, field_ap
        
        else:
            return eff_s
    
    def calcTaper(self, surfaceObject, field, R_aper):
        x = surfaceObject.grid_x
        y = surfaceObject.grid_y
        area = surfaceObject.area
        
        maskAper = np.sqrt(x**2 + y**2) < R_aper

        field_ap = field * maskAper.astype(complex)
        
        eff_t = np.absolute(np.sum(field[maskAper] * area[maskAper]))**2 / np.sum(np.absolute(field[maskAper])**2 * area[maskAper]) / np.sum(maskAper*area)
        
        return eff_t
