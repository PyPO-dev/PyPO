import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Python.Colormaps as cmaps

from src.Python.System import System as sy
from src.Python.Efficiencies import Efficiencies as eff

def makeASTE():
    # Secondary parameters
    d_foc           = 5606.286
    foc_2_h1        = np.array([0,0,3.5e3 -  d_foc])

    # Initialize system
    s = sy()
    
    #center_cam = foc_1_h1 + np.array([0,0,1e4])#foc_2_h1 # Place the camera at the z coordinate of the hyperbolic secondary focus
    center_cam = foc_2_h1 # Place camera at secondary focus
    lims_x_cam = [-65 - 50, 65 - 50]
    lims_y_cam = [-65, 65]
    gridsize_cam = [101, 101]
    
    # Add camera surface to optical system
    s.addCamera(name = "cam1", center=center_cam, lims_x=lims_x_cam, lims_y=lims_y_cam, gridsize=gridsize_cam)

    s.system["cam1"].setGrid(lims_x_cam, lims_y_cam, gridsize_cam)
    
    #s.plotSystem(focus_1=True, focus_2=True)
    
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
    
    # No chopper
    '''
    lims_x_beam = np.array([-71.9, 71.9])
    lims_y_beam = np.array([-71.9, 71.9])
    gridsize_beam = [101, 101]
    '''
    
    # pos 1 (+) & 2 (-)
    #'''
    lims_x_beam = np.array([-65, 65]) + 50
    lims_y_beam = np.array([-65, 65])
    gridsize_beam = [71, 71]
    #'''
    
    s.addBeam(lims_x=lims_x_beam, lims_y=lims_y_beam, gridsize=gridsize_beam, name=name+'.txt', beam='custom', flip=True)
    '''
    pt.imshow(20*np.log10(np.absolute(s.inputBeam.Ex) / np.max(np.absolute(s.inputBeam.Ex))), vmin=-30, vmax=0, origin='lower')
    pt.show()
    '''
    s.inputBeam.calcJM(mode='PMC')
    offTrans_pw = s.system["cam1"].center + np.array([0,0,250]) # Measurement plane
    s.inputBeam.transBeam(offTrans=offTrans_pw)

    s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11)
    s.runPhysOptics(save=2, material_source='vac')

    #s.PO.plotField(s.system["cam1"].grid_x, s.system["cam1"].grid_y, mode='Ex', show=True, title=r'$E_x$ in warm focal plane, $f$ = {} GHz'.format(name), polar=True)
    
    s.initFourierOptics(k)
    field_WF = s.loadField(s.system["cam1"], mode='Ex')
    gx_pad, gy_pad, field_pad = s.FO.padGrids(s.system["cam1"].grid_x, s.system["cam1"].grid_y, field_WF[0])
    
    gx = s.system["cam1"].grid_x
    gy = s.system["cam1"].grid_y
    
    xb = s.inputBeam.grid_x
    yb = s.inputBeam.grid_y
    
    kx, ky, APWS_pr = s.FO.propagateAPWS(xb, yb, s.inputBeam.Ex, z=-250, returns='APWS_a')
    
    tx, ty, APWS_WF = s.FO.propagateAPWS(gx_pad, gy_pad, field_pad, z=1, returns='APWS_a')
    
    field_ff_APWS = s.FO.ift2(APWS_pr)
    
    '''
    extent2 = [np.min(gx), np.max(gx), np.min(gy), np.max(gy)]
    
    extent = [np.min(tx), np.max(tx), np.min(ty), np.max(ty)]
    
    extentk = [np.min(xb), np.max(xb), np.min(yb), np.max(yb)]
    
    fig, ax = pt.subplots(3,2, figsize=(10,15))

    ax[0,0].imshow(20*np.log10(np.absolute(field_WF[0]) / np.max(np.absolute(field_WF[0]))), vmin=-30, vmax=0, origin='lower', extent=extent2)
    
    ax[0,1].imshow(np.angle(field_WF[0]), origin='lower', extent=extent2)
    
    ax[1,0].imshow(20*np.log10(np.absolute(APWS_WF) / np.max(np.absolute(APWS_WF))), vmin=-30, vmax=0, origin='lower', extent=extent)
    
    ax[1,1].imshow(np.angle(APWS_WF), origin='lower', extent=extent)

    ax[2,0].imshow(20*np.log10(np.absolute(field_ff_APWS) / np.max(np.absolute(field_ff_APWS))), vmin=-30, vmax=0, origin='lower', extent=extentk)
    
    ax[2,1].imshow(np.angle(field_ff_APWS), origin='lower', extent=extentk)

    ax[2,0].imshow(20*np.log10(np.absolute(APWS_pr) / np.max(np.absolute(APWS_pr))), vmin=-30, vmax=0, origin='lower', extent=extentk)
    
    ax[2,1].imshow(np.angle(APWS_pr), origin='lower', extent=extentk)

    pt.show()
    pt.close()
    '''
    
    R_pri = 5e3
    R_sec = 310
    
    scale_fac = R_pri / R_sec
    sec_ang = 3.226
    pri_ang = 71.075
    blk_ang = sec_ang / scale_fac
    
    M_p = 25.366
    f_pri = 3500 # mm
    
    f_sys = M_p * f_pri

    cond1 = np.sqrt(tx**2 + ty**2) < sec_ang
    cond2 = np.sqrt(tx**2 + ty**2) > blk_ang
    
    cond1_pr = np.sqrt(kx**2 + ky**2) < sec_ang
    cond2_pr = np.sqrt(kx**2 + ky**2) > blk_ang
    
    sec_range = cond1 & cond2
    sec_range_pr = cond1_pr & cond2_pr
    
    APWS_trunc = APWS_WF * sec_range.astype(complex)
    APWS_trunc_pr = APWS_pr * sec_range_pr.astype(complex)
    
    az = np.degrees(gx_pad / f_sys) * 3600 # arcsec
    el = np.degrees(gy_pad / f_sys) * 3600 # arcsec
    
    azb = np.degrees(xb / f_sys) * 3600 # arcsec
    elb = np.degrees(yb / f_sys) * 3600 # arcsec
    
    ff = s.FO.ift2(APWS_trunc)
    ff_pr = s.FO.ift2(APWS_trunc_pr)
    '''
    # Plot far-field
    extent = [np.min(az), np.max(az), np.min(el), np.max(el)]
    extentb = [np.min(azb), np.max(azb), np.min(elb), np.max(elb)]
    
    fig, ax = pt.subplots(1,2, figsize=(10,5))
    amp = ax[1].imshow(20*np.log10(np.absolute(ff) / np.max(np.absolute(ff))), vmin=-30, vmax=0, origin='lower', extent=extent, cmap=cmaps.parula)
    
    ax[0].imshow(20*np.log10(np.absolute(ff_pr) / np.max(np.absolute(ff_pr))), vmin=-30, vmax=0, origin='lower', extent=extentb, cmap=cmaps.parula)
    
    divider = make_axes_locatable(ax[1])

    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    ax[1].set_xlim(-300, 300)
    ax[1].set_ylim(-300, 300)
        
    ax[1].set_ylabel(r"El / [as]")
    ax[1].set_xlabel(r"Az / [as]")
    c = fig.colorbar(amp, cax=cax, orientation='vertical')
    
    ax[1].set_title(r'$f$ = {} GHz'.format(name))
    
    #pt.show()
    #pt.savefig(fname='personal/pointing/pos_2/{}.jpg'.format(name),bbox_inches='tight', dpi=300)
    #pt.close()
    '''
    maxIdx = np.argmax(np.absolute(ff_pr.T))
    maxIdx = np.unravel_index(maxIdx, ff_pr.shape)
    
    az_ff = azb[maxIdx]
    el_ff = elb[maxIdx]
    
    print(az_ff)
    print(el_ff)
    
    return az_ff, el_ff
    
if __name__ == "__main__":
    cb = './custom/beam/pos_1/'
    
    name = ['168', '178', '188', '198', '208', '218', '228', '238', '247', '257', '267', '277', '287', '297', '307', '317', '327', '337', '346', '356', '366', '376', '386']
    
    np.save('freq_range.npy', name)
    
    az = []
    el = []
    #name = ['238']
    for na in name:
    
        _az, _el = main(cb, na)
        az.append(_az)
        el.append(_el)
        
    np.save(cb + 'az_APWS.npy', az)
    np.save(cb + 'el_APWS.npy', el)
 
