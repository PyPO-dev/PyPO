import numpy as np
from src.Python.System import System
def buildSystem():
    s = System()
    R_vert = 0

    # [mm], vertex hole radius
    R_aper = 12.5e3 # [mm], aperture radius
    R_focs = 12e3
    # [mm], focal distance
    lix = [R_vert, R_aper]
    liy = [0, 2 * np.pi]
    vert = np.zeros(3)
    focs = np.array([0, 0, R_focs])
    coefs = [focs, vert]
    gridsize = [201, 201]
    s.addParabola(coefs, lix, liy, gridsize, pmode='foc', gmode='uv',
    name='DRO_pri')
    s.addPlotter()
    s.plotter.plot3D(s.system['DRO_pri'], foc1=True)
    
if __name__ == '__main__':
    buildSystem()