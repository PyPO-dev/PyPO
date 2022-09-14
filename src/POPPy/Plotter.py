import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.POPPy.Colormaps as cmaps

pt.rcParams['xtick.top'] = True
pt.rcParams['ytick.right'] = True

pt.rcParams['xtick.direction'] = "in"
pt.rcParams['ytick.direction'] = "in"

class Plotter(object):
    """
    Class for plotting stuff. Can plot beams n stuff
    """
    def __init__(self, save):
        self.savePath = save
        
        existSave = os.path.isdir(self.savePath)
        
        if not existSave:
            os.makedirs(self.savePath)
    
    def plotBeam2D(self, surfaceObject, field, vmin=-30, vmax=0, show=True, amp_only=False, save=False, interpolation=None, mode='dB', units=''):
        comp = field[1]
        field = field[0]
        
        titleAmp = r"${}_{}$ Amp / [dB]".format(comp[0], comp[1])
        titlePhase = r"${}_{}$ Phase / [rad]".format(comp[0], comp[1])
        
        # Obtain conversion units if manually given
        if units:
            conv = surfaceObject.get_conv(units)
            units = units
        else:
            conv = surfaceObject.conv
            units = surfaceObject.units

        if surfaceObject.elType == "Camera":
            if surfaceObject.ff_flag:
                grid_x1 = surfaceObject.grid_Az / conv
                grid_x2 = surfaceObject.grid_El / conv
                
                ff_flag = True
            
        else:
            grid_x1 = surfaceObject.grid_x / conv
            grid_x2 = surfaceObject.grid_y / conv
            
            ff_flag = False

        extent = [grid_x1[0,0], grid_x1[-1,0], grid_x2[0,0], grid_x2[0,-1]]
    
        if not amp_only:
            fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})
        
            divider1 = make_axes_locatable(ax[0])
            divider2 = make_axes_locatable(ax[1])

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
        
            if mode == 'linear':
                ampfig = ax[0].imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            elif mode == 'dB':
                ampfig = ax[0].imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)
            
            if ff_flag:
                ax[0].set_ylabel(r"El / [{}]".format(units))
                ax[0].set_xlabel(r"Az / [{}]".format(units))
                ax[1].set_ylabel(r"El / [{}]".format(units))
                ax[1].set_xlabel(r"Az / [{}]".format(units))
            
            else:
                ax[0].set_ylabel(r"$y$ / [{}]".format(units))
                ax[0].set_xlabel(r"$x$ / [{}]".format(units))
                ax[1].set_ylabel(r"$y$ / [{}]".format(units))
                ax[1].set_xlabel(r"$x$ / [{}]".format(units))
        
            ax[0].set_title(titleAmp, y=1.08)
            ax[0].set_aspect(1)
            ax[1].set_title(titlePhase, y=1.08)
            ax[1].set_aspect(1)
        
            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
            c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')
            
        else:
            fig, ax = pt.subplots(1,1, figsize=(5,5))
        
            divider = make_axes_locatable(ax)

            cax = divider.append_axes('right', size='5%', pad=0.05)

            if mode == 'linear':
                ampfig = ax.imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula)
            
            elif mode == 'dB':
                ampfig = ax.imshow(20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
            
            if ff:
                ax.set_ylabel(r"El / [{}]".format(units))
                ax.set_xlabel(r"Az / [{}]".format(units))
            
            else:
                ax.set_ylabel(r"$y$ / [{}]".format(conv))
                ax.set_xlabel(r"$x$ / [{}]".format(conv))
        
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
        
    def beamCut(self, plotObject, field, cross='', units='', vmin=-50, vmax=0, frac=1, show=True, save=False):
        if units:
            conv = plotObject.get_conv(units)
            units = units
        else:
            conv = plotObject.conv
            units = plotObject.units
        
        x_center = int((plotObject.shape[0] - 1) / 2)
        y_center = int((plotObject.shape[1] - 1) / 2)

        comp = field[1]
        field = field[0]
        
        field_dB = 20 * np.log10(np.absolute(field) / np.max(np.absolute(field)))
        
        H_cut = field_dB[x_center,:]
        E_cut = field_dB[:,y_center]
        
        # TODO: make this nicer
        x_ax = plotObject.grid_Az[:,0] / conv
        
        fig, ax = pt.subplots(1,1, figsize=(7,5))
        
        ax.plot(x_ax, H_cut, color='blue', label='H-plane')
        ax.plot(x_ax, E_cut, color='red', ls='--', label='E-plane')
        
        if cross:
            diago = np.diag(20 * np.log10(np.absolute(cross[0]) / np.max(np.absolute(field))))
            print(diago.shape)
            ax.plot(x_ax, diago, color='purple', ls='dashdot', label='X-pol')
        
        ax.set_xlabel(r'$\theta$ / [{}]'.format(units))
        ax.set_ylabel(r'Amplitude / [dB]')
        ax.set_title('Beam cuts')
        ax.set_box_aspect(1)
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(x_ax[0], x_ax[-1])
        ax.legend(frameon=False, prop={'size': 10},handlelength=1)

        if save:
            pt.savefig(fname=self.savePath + 'EH_cut_{}.jpg'.format(plotObject.name),bbox_inches='tight', dpi=300)
            
        if show:
            pt.show()
        
        pt.close()
        
        diff_abs = np.absolute(field[:,y_center]) - np.absolute(field[x_center,:])
        diff_ang = np.angle(field[:,y_center]) - np.angle(field[x_center,:])
        
        fig, ax = pt.subplots(1,3)
        ax[0].plot(np.absolute(field[:,y_center]))
        ax[0].plot(np.absolute(field[x_center,:]))
        ax[1].plot(diff_abs)
        ax[2].plot(diff_ang)
        pt.show()
        
        
        
        
        
        
        
