import numpy as np
import matplotlib.pyplot as pt

class CoordConvert(object):
    def wavevecToSpherical(self, kx, ky, k):
        
        _kz_im = k**2 - kx**2 - ky**2
        _kz_im[_kz_im < 0] = 0
        
        kz = np.sqrt(_kz_im)
        '''
        theta = np.arccos(kz / k)
        phi = np.arccos(kx / (k * np.sin(theta)))
        '''
        sumxy = np.sqrt(kx**2 + ky**2)
        #sumxy[sumxy == 0] = kz[sumxy == 0]

        theta = np.arccos(kz / k)
        phi = np.arctan2(ky, kx)
        
        return theta, phi
    
    def cartToSphericalComp(self, fieldx, fieldy, fieldz, theta, phi):
        f_t = list(fieldx[1])[0]
        
        field_th = np.cos(theta)*np.cos(phi)*fieldx[0] + np.cos(theta)*np.sin(phi)*fieldy[0] - np.sin(theta)*fieldz[0]
        
        field_ph = -np.sin(phi)*fieldx[0] + np.cos(phi)*fieldy[0]
        
        cth = f_t + "th"
        cph = f_t + "ph"
        
        return [field_th, cth], [field_ph, cph]
        
