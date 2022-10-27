import numpy as np
import matplotlib.pyplot as pt

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
    
    def calcTaperfromPriFocus(self, grid_th, grid_ph, field, mask):
        """
        (PUBLIC)
        Calculate taper efficiency for primary focus illumination.
        """
        field = field[0]
        field_ap = field[mask]
        
        nomi = np.absolute(np.sum(field_ap))**2
        deno = np.sum(np.absolute(field_ap)**2) * np.sum(mask**2)

        eta_t = nomi / deno 

        return eta_t
    
    def calcPhaseEff(self, field, mask):
        field = field[0]
        
        to_calc = field[mask]
        pe_rms = np.mean(np.angle(to_calc))
        m = np.max(np.absolute(np.angle(to_calc) - pe_rms))
        
        eta_ph = (1 - m**2/2)**2
        return eta_ph
    
    def calcSpilloverfromPriFocus(self, grid_th, grid_ph, field, mask):
        """
        (PUBLIC)
        Calculate taper efficiency for primary focus illumination.
        """
        field = field[0]
        field_ap = field * mask.astype(complex)

        nomi = np.absolute(np.sum(np.conj(field_ap) * field))**2
        deno = np.sum(np.absolute(field)**2) * np.sum(np.absolute(field_ap)**2)

        eta_s = nomi / deno 

        return eta_s
    
    def calcTaperfromCassFocus(self, grid_th, grid_ph, field, f_pri, M, R_pri):
        f = f_pri * M
        
        th0 = np.degrees(2 * np.arctan(1 / (2*f/R_pri)))
        
        mask = np.sqrt(grid_th**2 + grid_ph**2) < th0

        eta_t = self.calcTaperfromPriFocus(grid_th, grid_ph, field, mask)
        
        return eta_t
    
    def calcSpilloverfromCassFocus(self, grid_th, grid_ph, field, f_pri, M, R_pri):
        f = f_pri * M
        
        th0 = np.degrees(2 * np.arctan(1 / (2*f/R_pri)))
        
        mask = np.sqrt(grid_th**2 + grid_ph**2) < th0

        eta_s = self.calcSpilloverfromPriFocus(grid_th, grid_ph, field, mask)
        
        return eta_s
    
    def calcPhaseEfffromCassFocus(self, grid_th, grid_ph, field, f_pri, M, R_pri):
        f = f_pri * M
        
        th0 = np.degrees(2 * np.arctan(1 / (2*f/R_pri)))
        
        mask = np.sqrt(grid_th**2 + grid_ph**2) < th0
        
        eta_ph = self.calcPhaseEff(field, mask)
        
        return eta_ph
