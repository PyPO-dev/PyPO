import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import src.POPPy.Colormaps as cmaps
from src.POPPy.BindRefl import *

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

    def plotBeam2D(self, surfaceObject, field, vmin=-30, vmax=0, show=True, amp_only=False, save=False, polar=False, interpolation=None, plot_aper={"plot":False}, mode='dB', project='xy', units='', name=''):
        comp = field[1]
        field = field[0]

        titleAmp = r"${}_{}$ Amp / [dB]".format(comp[0], comp[1])
        titlePhase = r"${}_{}$ Phase / [rad]".format(comp[0], comp[1])

        # Obtain conversion units if manually given
        if units:
            conv = surfaceObject._get_conv(units)
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
                ff_flag = False

        if project == 'xy':
            grid_x1 = surfaceObject.grid_x / conv
            grid_x2 = surfaceObject.grid_y / conv

            cx1 = 'x'
            cx2 = 'y'

            ff_flag = False


        elif project == 'yz':
            grid_x1 = surfaceObject.grid_y / conv
            grid_x2 = surfaceObject.grid_z / conv

            cx1 = 'y'
            cx2 = 'z'

            ff_flag = False

        elif project == 'zx':
            grid_x1 = surfaceObject.grid_z / conv
            grid_x2 = surfaceObject.grid_x / conv

            cx1 = 'z'
            cx2 = 'x'

            ff_flag = False

        extent = [np.min(grid_x1), np.max(grid_x1), np.min(grid_x2), np.max(grid_x2)]

        if not amp_only:
            fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})

            divider1 = make_axes_locatable(ax[0])
            divider2 = make_axes_locatable(ax[1])

            cax1 = divider1.append_axes('right', size='5%', pad=0.05)
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)

            if mode == 'linear':
                ampfig = ax[0].imshow(np.absolute(field), origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)
                phasefig = ax[1].imshow(np.angle(field), origin='lower', extent=extent, cmap=cmaps.parula)

            elif mode == 'dB':
                if polar:
                    extent = [grid_x1[0,0], grid_x1[-1,0], grid_x2[0,0], grid_x2[0,-1]]
                    ampfig = ax[0].pcolormesh(grid_x1, grid_x2, 20 * np.log10(np.absolute(field) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
                    phasefig = ax[1].pcolormesh(grid_x1, grid_x2, np.angle(field), cmap=cmaps.parula, shading='auto')

                else:
                    phasefig = ax[1].imshow(np.angle(np.flip(field)), origin='lower', extent=extent, cmap=cmaps.parula)
                    ampfig = ax[0].imshow(20 * np.log10(np.absolute(np.flip(field)) / np.max(np.absolute(field))), vmin=vmin, vmax=vmax, origin='lower', extent=extent, cmap=cmaps.parula, interpolation=interpolation)


            if ff_flag:
                ax[0].set_ylabel(r"El / [{}]".format(units))
                ax[0].set_xlabel(r"Az / [{}]".format(units))
                ax[1].set_ylabel(r"El / [{}]".format(units))
                ax[1].set_xlabel(r"Az / [{}]".format(units))

            else:
                ax[0].set_ylabel(r"${}$ / [{}]".format(cx2, units))
                ax[0].set_xlabel(r"${}$ / [{}]".format(cx1, units))
                ax[1].set_ylabel(r"${}$ / [{}]".format(cx2, units))
                ax[1].set_xlabel(r"${}$ / [{}]".format(cx1, units))

            ax[0].set_title(titleAmp, y=1.08)
            ax[0].set_aspect(1)
            ax[1].set_title(titlePhase, y=1.08)
            ax[1].set_aspect(1)

            c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
            c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')

            if plot_aper["plot"]:
                xc = plot_aper["center"][0]
                yc = plot_aper["center"][1]
                R = plot_aper["radius"]

                circle=mpl.patches.Circle((xc,yc),R, color='black', fill=False)

                ax[0].add_patch(circle)
                ax[0].scatter(xc, yc, color='black', marker='x')

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
                ax.set_ylabel(r"${}$ / [{}]".format(cx2, units))
                ax.set_xlabel(r"${}$ / [{}]".format(cx1, units))

            ax.set_title(titleAmp, y=1.08)
            ax.set_box_aspect(1)

            c = fig.colorbar(ampfig, cax=cax, orientation='vertical')

            if plot_aper["plot"]:
                xc = plot_aper["center"][0]
                yc = plot_aper["center"][1]
                R = plot_aper["radius"]

                circle=mpl.patches.Circle((xc,yc),R, color='black', fill=False)

                ax.add_patch(circle)
                ax.scatter(xc, yc, color='black', marker='x')


        if save:
            pt.savefig(fname=self.savePath + '{}_{}_{}.jpg'.format(surfaceObject.name, comp, name),bbox_inches='tight', dpi=300)

        if show:
            pt.show()

        pt.close()

    def plot3D(self, plotObject, fine=2, cmap=cm.cool,
                returns=False, ax_append=False, norm=False,
                show=True, foc1=False, foc2=False, save=True):
        skip = slice(None,None,fine)
        grids = generateGrid(plotObject)

        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        reflector = ax_append.plot_surface(grids.x[skip], grids.y[skip], grids.z[skip],
                       linewidth=0, antialiased=False, alpha=0.5, cmap=cmap)

        if foc1:
            ax_append.scatter(plotObject["focus_1"][0], plotObject["focus_1"][1], plotObject["focus_1"][2], color='black')

        if foc2:
            ax_append.scatter(plotObject["focus_2"][0], plotObject["focus_2"][1], plotObject["focus_2"][2], color='black')

        if norm:
            length = np.sqrt(np.dot(plotObject["focus_1"], plotObject["focus_1"])) / 5
            skipn = slice(None,None,10*fine)
            ax_append.quiver(grids.x[skipn,skipn], grids.y[skipn,skipn], grids.z[skipn,skipn],
                            grids.nx[skipn,skipn], grids.ny[skipn,skipn], grids.nz[skipn,skipn],
                            color='black', length=length, normalize=True)
            print(np.sqrt(grids.nx**2 + grids.ny**2 + grids.nz**2))
        if not returns:
            ax_append.set_ylabel(r"$y$ / [mm]", labelpad=20)
            ax_append.set_xlabel(r"$x$ / [mm]", labelpad=10)
            ax_append.set_zlabel(r"$z$ / [mm]", labelpad=50)
            ax_append.set_title(plotObject["name"], fontsize=20)
            world_limits = ax_append.get_w_lims()
            ax_append.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
            ax_append.tick_params(axis='x', which='major', pad=-3)

        if save:
            pt.savefig(fname=self.savePath + '{}.jpg'.format(plotObject["name"]),bbox_inches='tight', dpi=300)

        if show:
            pt.show()

        if returns:
            return ax_append

        pt.close()

    def plotSystem(self, systemDict, fine=2, cmap=cm.cool,
                ax_append=False, norm=False,
                show=True, foc1=False, foc2=False, save=True, ret=False):

        fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})

        for key, refl in systemDict.items():
            self.plot3D(plotObject=refl, fine=fine, cmap=cmap,
                        returns=True, ax_append=ax, norm=norm,
                        show=False, foc1=foc1, foc2=foc2, save=False)

        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        ax.set_title("System", fontsize=20)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)

        if save:
            pt.savefig(fname=self.savePath + 'system.jpg',bbox_inches='tight', dpi=300)

        if show:
            pt.show()

        if ret:
            return fig, ax

        pt.close()

    def beamCut(self, plotObject, field, cross='', units='', vmin=-50, vmax=0, frac=1, show=True, save=False, ret=False):
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
        if surfaceObject.ff_flag:
            x_ax = plotObject.grid_Az[:,0] / conv

        elif surfaceObject.fo_flag:
            x_ax = plotObject.grid_fox[:,0]

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

        if ret:
            return field[:,y_center], field[:,y_center]
