import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.Python.Colormaps as cmaps

class Plotter(object):
    """
    Class for plotting stuff. Can plot beams n stuff
    """
    def __init__(self, save):
        self.savePath = save
        
        existSave = os.path.isdir(self.savePath)
        
        if not existSave:
            os.makedirs(self.savePath)
    
    def plotBeam2D(self, surfaceObject, field, vmin=-30, vmax=0, ff=0, show=True, amp_only=False, save=True, interpolation=None, mode='dB'):
        
        titleAmp = "PNA / [dB]"
        titlePhase = "Phase / [rad]"
        
        comp = field[1]
        field = field[0]
        
        if ff:
            extent = [surfaceObject.grid_x[0,0] / ff * 3600, surfaceObject.grid_x[-1,0] / ff * 3600, surfaceObject.grid_y[0,0] / ff * 3600, surfaceObject.grid_y[0,-1] / ff * 3600]
            
        else:
            extent = [surfaceObject.grid_x[0,0], surfaceObject.grid_x[-1,0], surfaceObject.grid_y[0,0], surfaceObject.grid_y[0,-1]]
        
        
        if not amp_only:
            fig, ax = pt.subplots(1,2)
        
            divider1 = make_axes_locatable(ax[0])
            divider2 = make_axes_locatable(ax[1])

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
            if mode == 'linear':
                ampfig = ax[0].imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            elif mode == 'dB':
                ampfig = ax[0].imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)
            
            if ff:
                ax[0].set_ylabel(r"El / [as]")
                ax[0].set_xlabel(r"Az / [as]")
                ax[1].set_ylabel(r"El / [as]")
                ax[1].set_xlabel(r"Az / [as]")
            
            else:
                ax[0].set_ylabel(r"$y$ / [mm]")
                ax[0].set_xlabel(r"$x$ / [mm]")
                ax[1].set_ylabel(r"$y$ / [mm]")
                ax[1].set_xlabel(r"$x$ / [mm]")
        
            ax[0].set_title(titleAmp, y=1.08)
            ax[0].set_aspect(1)
            ax[1].set_title(titlePhase, y=1.08)
            ax[1].set_aspect(1)
        
            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
            c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
            
        else:
            fig, ax = pt.subplots(1,1)
        
            divider = make_axes_locatable(ax)

            cax = divider.append_axes('right', size='5%', pad=0.05)

            if mode == 'linear':
                ampfig = ax.imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula)
            
            elif mode == 'dB':
                ampfig = ax.imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            if ff:
                ax.set_ylabel(r"El / [as]")
                ax.set_xlabel(r"Az / [as]")
            
            else:
                ax.set_ylabel(r"$y$ / [mm]")
                ax.set_xlabel(r"$x$ / [mm]")
        
            ax.set_title(titleAmp, y=1.08)
            ax.set_box_aspect(1)
        
            c = fig.colorbar(ampfig, cax=cax, orientation='vertical')
        
        if save:
            pt.savefig(fname=self.savePath + '{}_{}.jpg'.format(surfaceObject.name, comp),bbox_inches='tight', dpi=300)
        
        if show:
            pt.show()
            
        pt.close()
        
    def plot3D(self, plotObject, fine=2, cmap=cm.cool, returns=False, ax_append=False, norm=False, show=True, foc1=False, foc2=False, save=True):
        skip = slice(None,None,fine)
        
        length = np.sqrt(np.dot(plotObject.focus_1, plotObject.focus_1)) / 5
        
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        reflector = ax_append.plot_surface(plotObject.grid_x[skip], plotObject.grid_y[skip], plotObject.grid_z[skip],
                       linewidth=0, antialiased=False, alpha=0.5, cmap=cmap)
        
        if foc1:
            ax_append.scatter(plotObject.focus_1[0], plotObject.focus_1[1], plotObject.focus_1[2], color='black')
            
        if foc2:
            ax_append.scatter(plotObject.focus_2[0], plotObject.focus_2[1], plotObject.focus_2[2], color='black')
            
        if norm:
            skipn = slice(None,None,10*fine)
            ax_append.quiver(plotObject.grid_x[skipn,skipn], plotObject.grid_y[skipn,skipn], plotObject.grid_z[skipn,skipn], plotObject.grid_nx[skipn,skipn], plotObject.grid_ny[skipn,skipn], plotObject.grid_nz[skipn,skipn], color='black', length=length, normalize=True)

        if not returns:
            ax_append.set_ylabel(r"$y$ / [mm]", labelpad=20)
            ax_append.set_xlabel(r"$x$ / [mm]", labelpad=10)
            ax_append.set_zlabel(r"$z$ / [mm]", labelpad=50)
            ax_append.set_title(plotObject.name, fontsize=20)
            world_limits = ax_append.get_w_lims()
            ax_append.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
            ax_append.tick_params(axis='x', which='major', pad=-3)
            
        if save:
            pt.savefig(fname=self.savePath + '{}.jpg'.format(plotObject.name),bbox_inches='tight', dpi=300)
            
        if show:
            pt.show()
        
        pt.close()
         
        if returns:
            return ax_append
