import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Python.Colormaps as cmaps

class Plotter(object):
    """
    Class for plotting stuff. Can plot beams n stuff
    """
    
    def plotBeam2D(self, grid_x, grid_y, field, mode='dB', vmin=-30, vmax=0, show=True, amp_only=False, save=0, titleAmp="PNA / [dB]", titlePhase="Phase / [rad]", interpolation=None):
        if not amp_only:
            fig, ax = pt.subplots(1,2)
        
            divider1 = make_axes_locatable(ax[0])
            divider2 = make_axes_locatable(ax[1])

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
            extent = [grid_x[0,0], grid_x[-1,0], grid_y[0,0], grid_y[0,-1]]
        
            if mode == 'linear':
                ampfig = ax[0].imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            elif mode == 'dB':
                ampfig = ax[0].imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)
        
            ax[0].set_ylabel(r"$y$ / [mm]")
            ax[0].set_xlabel(r"$x$ / [mm]")
            ax[1].set_ylabel(r"$y$ / [mm]")
            ax[1].set_xlabel(r"$x$ / [mm]")
        
            ax[0].set_title(titleAmp, y=1.08)
            ax[0].set_box_aspect(1)
            ax[1].set_title(titlePhase, y=1.08)
            ax[1].set_box_aspect(1)
        
            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
            c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
            
        else:
            fig, ax = pt.subplots(1,1)
        
            divider = make_axes_locatable(ax)

            cax = divider.append_axes('right', size='5%', pad=0.05)
        
            extent = [grid_x[0,0], grid_x[-1,0], grid_y[0,0], grid_y[0,-1]]
        
            if mode == 'linear':
                ampfig = ax.imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula)
            
            elif mode == 'dB':
                ampfig = ax.imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)

            ax.set_ylabel(r"$y$ / [mm]")
            ax.set_xlabel(r"$x$ / [mm]")
        
            ax.set_title(titleAmp, y=1.08)
            ax.set_box_aspect(1)
        
            c = fig.colorbar(ampfig, cax=cax, orientation='vertical')
        
        if show:
            pt.show()
            
        if save:
            pt.savefig(fname="{}.jpg".format(save),bbox_inches='tight', dpi=300)
            
        pt.close()
