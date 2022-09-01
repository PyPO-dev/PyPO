import numpy as np

from src.Python.System import System as sy
from src.Python.Efficiencies import Efficiencies as eff

def makeASTE():
    # Primary parameters
    R_pri           = 5e3 # Radius in [mm]
    R_aper          = 300#200 # Vertex hole radius in [mm]
    foc_pri         = np.array([0,0,3.5e3]) # Coordinates of focal point in [mm]
    ver_pri         = np.zeros(3) # Coordinates of vertex point in [mm]
    
    # Pack coefficients together for instantiating parabola: [focus, vertex]
    coef_p1         = [foc_pri, ver_pri]
    gridsize_p1     = [201, 201] # The gridsizes along the u and v axes
    
    lims_r_p1       = [R_aper, R_pri]
    lims_v_p1       = [0, 2*np.pi]
    
    # Secondary parameters
    R_sec           = 310
    d_foc           = 5606.286
    foc_1_h1        = np.array([0,0,3.5e3])
    foc_2_h1        = np.array([0,0,3.5e3 -  d_foc])
    ecc_h1          =  1.08208248
    
    # Pack coefficients together for instantiating hyperbola: [focus 1, focus 2, eccentricity]
    coef_h1         = [foc_1_h1, foc_2_h1, ecc_h1]
    gridsize_h1     = [501, 501]
    
    lims_r_h1       = [0, R_sec]
    lims_v_h1       = [0, 2*np.pi]

    # Initialize system
    s = sy()
    
    # Add parabolic reflector and hyperbolic reflector by focus, vertex and two foci and eccentricity
    s.addParabola(name = "p1", coef=coef_p1, lims_x=lims_r_p1, lims_y=lims_v_p1, gridsize=gridsize_p1, pmode='foc', gmode='uv')
    s.addHyperbola(name = "h1", coef=coef_h1, lims_x=lims_r_h1, lims_y=lims_v_h1, gridsize=gridsize_h1, pmode='foc', gmode='uv')

    # Instantiate camera surface. Size does not matter, as long as z coordinate agrees
    h_hyp = np.max(s.system["h1"].grid_z)
    
    #center_cam = foc_1_h1 + np.array([0,0,1e4])#foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    center_cam = np.array([0, 0, h_hyp]) # Place camera at secondary aperture
    lims_x_cam = [-1000, 1000]
    lims_y_cam = [-1000, 1000]
    gridsize_cam = [201, 201]
    
    # Add camera surface to optical system
    s.addCamera(name = "cam1", center=center_cam, lims_x=lims_x_cam, lims_y=lims_y_cam, gridsize=gridsize_cam)
    
    #print(s.system["p1"])
    #print(s.system["h1"])
    #print(s.system["cam1"])

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    #s.system["cam1"].plotCamera()
    
    #s.plotSystem(focus_1=True, focus_2=True)
    
    r_max_vertex = 300
    off_vertex = np.array([0,0,2106.286])
    
    gridsize_aper = [201, 201]
    
    s.addCircAper(r_max_vertex, gridsize_aper, name='vertex')
    print(s.system)
    
    s.plotSystem(focus_1=True, focus_2=True)
    
    return s

def main(cb, name):
    """
    Script for PO propagation of measured beams at measurement plane to ASTE far field.
    First, define ASTE cassegrain.
    Then, propagate through setup into aperture plane.
    Then, use Fraunhofer far-field to calculate far-field.
    """
    s = makeASTE()
    int_na = int(name)
    int_na *= 1e9
    '''
    beami = np.load(cb + name + '.npy')
    np.savetxt(cb + 'r' + name + '.txt', np.real(beami))
    np.savetxt(cb + 'i' + name + '.txt', np.imag(beami))
    '''
    cl = 3e8 * 1e3
    
    lam = cl / int_na
    
    print(lam)
    
    k = 2 * np.pi / lam # mm
    
    s.setCustomBeamPath(cb)
    s.addPlotter()
    
    # Uncorrected
    '''
    lims_x_beam = np.array([-71.9, 71.9]) - 12
    lims_y_beam = np.array([-71.9, 71.9]) + 2
    gridsize_beam = [101, 101]
    '''
    
    # No chopper
    #'''
    lims_x_beam = np.array([-71.9, 71.9])
    lims_y_beam = np.array([-71.9, 71.9])
    gridsize_beam = [101, 101]
    #'''
    
    # pos 1 (+) & 2 (-)
    '''
    lims_x_beam = np.array([-65, 65]) - 50
    lims_y_beam = np.array([-65, 65])
    gridsize_beam = [71, 71]
    '''
    
    s.addBeam(lims_x=lims_x_beam, lims_y=lims_y_beam, gridsize=gridsize_beam, name=name+'.txt', beam='custom')
    
    #print(s.inputBeam.Ex.shape)
    
    s.inputBeam.calcJM(mode='PMC')
    #s.plotter.plotBeam2D(s.inputBeam, [s.inputBeam.Ex, 'Ex'])
    offTrans_pw = s.system["h1"].focus_2 + np.array([0,0,250]) # Measurement plane
    s.inputBeam.transBeam(offTrans=offTrans_pw)
    
    #s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11)
    s.initPhysOptics(target=s.system["vertex"], k=k, numThreads=11)
    #s.initPhysOptics(target=s.system["h1"], k=k, numThreads=11)
    s.runPhysOptics(save=2, material_target='vac')
    
    #print(np.min(s.system["vertex"].grid_y))
    
    #s.PO.plotField(s.system["h1"].grid_x, s.system["h1"].grid_y, mode='Ex', show=True, title=r'$E_x$ at secondary plane, $f$ = {} GHz'.format(name), polar=True)
    
    s.PO.plotField(s.system["vertex"].grid_x, s.system["vertex"].grid_y, mode='Ex', show=True, title=r'$E_x$ over secondary, $f$ = {} GHz'.format(name), polar=True)
    
    #s.PO.plotField(s.system["h1"].grid_x, s.system["h1"].grid_y, mode='Ex', show=True, title=r'$E_x$ at vertex plane, $f$ = {} GHz'.format(name), polar=True)
    
    s.nextPhysOptics(source=s.system["vertex"], target=s.system["cam1"])
    #s.nextPhysOptics(source=s.system["h1"], target=s.system["p1"])
    s.runPhysOptics(save=2, material_target='vac')
    
    #s.PO.plotField(s.system["p1"].grid_x, s.system["p1"].grid_y, mode='Ey', show=True, title=r'$E_x$ over parabola, $f$ = {} GHz'.format(name), polar=True)
    
    #s.nextPhysOptics(source=s.system["p1"], target=s.system["cam1"])
    #s.runPhysOptics(save=2)
    
    aperture = np.sqrt((s.system["cam1"].grid_x)**2 + (s.system["cam1"].grid_y)**2) < 310
    

    e = eff()
    field = s.loadField(s.system["cam1"], mode='Ex')
    spillover = e.calcSpillover(s.system["cam1"], field[0], aperture)
    taper = e.calcTaper(s.system["cam1"], field[0], aperture)
    print(spillover)
    print(taper)

    #s.plotSystem(focus_1=False, focus_2=False)#, exclude=[0,1,2])

    #s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ez')
    #s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ex', show=False, save='./personal/uncor/{}'.format(name), title=r'$E_x$ at secondary plane, $f$ = {} GHz'.format(name))
    
    s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ex', show=True, title=r'$E_x$ at secondary plane, $f$ = {} GHz'.format(name))
    
    return spillover
    
if __name__ == "__main__":
    cb = './custom/beam/no_chop/'
    
    name = ['168', '178', '188', '198', '208', '218', '228', '238', '247', '257', '267', '277', '287', '297', '307', '317', '327', '337', '346', '356', '366', '376', '386']
    
    eff_s = []
    
    na = '238'
    
    _eff_s = main(cb, na)
    for na in name:
    
        _eff_s = main(cb, na)
        eff_s.append(_eff_s)
        
    np.save(cb + 'eff_s_PO_sec.npy', eff_s)
    np.save(cb + 'frequency_range_PO_sec.npy', name)
