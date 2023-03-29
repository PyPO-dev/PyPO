import numpy as np
import os
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

import src.PyPO.PlotConfig
import src.PyPO.Colormaps as cmaps
from src.PyPO.BindRefl import *

##
# @file
# File containing functions for generating plots.

##
# Generate a 2D plot of a field or current.
#
# @param plotObject A reflDict containing surface on which to plot beam. 
# @param field PyPO field or current component to plot.
# @param vmin Minimum amplitude value to display. Default is -30.
# @param vmax Maximum amplitude value to display. Default is 0.
# @param show Show plot. Default is True.
# @param amp_only Only plot amplitude pattern. Default is False.
# @param save Save plot to /images/ folder.
# @param interpolation What interpolation to use for displaying amplitude pattern. Default is None.
# @param norm Normalise field (only relevant when plotting linear scale).
# @param aperDict Plot an aperture defined in an aperDict object along with the field or current patterns. Default is None.
# @param mode Plot amplitude in decibels ("dB") or on a linear scale ("linear"). Default is "dB".
# @param project Set abscissa and ordinate of plot. Should be given as a string. Default is "xy".
# @param units The units of the axes. Default is "", which is millimeters.
# @param name Name of .png file where plot is saved. Only when save=True. Default is "".
# @param titleA Title of the amplitude plot. Default is "Amp".
# @param titleP Title of the phase plot. Default is "Phase".
# @param savePath Path where plot will be saved if save = True.
# @param unwrap_phase Unwrap the phase patter. Prevents annular structure in phase pattern. Default is False.
#
# @returns fig Figure object containing plot.
# @returns ax Axes containing the axes of the plot.
#
# @see aperDict
def plotBeam2D(plotObject, field,
                vmin, vmax, show, amp_only,
                save, interpolation, norm,
                aperDict, mode, project,
                units, name, titleA, titleP, savePath, unwrap_phase):

    # With far-field, generate grid without converting to spherical
    max_field = np.max(np.absolute(field))
    grids = generateGrid(plotObject, transform=True, spheric=False)
    if not plotObject["gmode"] == 2:
        if project == 'xy':
            grid_x1 = grids.x
            grid_x2 = grids.y
            ff_flag = False

        elif project == 'yz':
            grid_x1 = grids.y
            grid_x2 = grids.z
            ff_flag = False

        elif project == 'zx':
            grid_x1 = grids.z
            grid_x2 = grids.x
            ff_flag = False

        elif project == 'yx':
            grid_x1 = grids.y
            grid_x2 = grids.x
            ff_flag = False

        elif project == 'zy':
            grid_x1 = grids.z
            grid_x2 = grids.y
            ff_flag = False

        elif project == 'xz':
            grid_x1 = grids.x
            grid_x2 = grids.z
            ff_flag = False

    else:
        if project == 'xy':
            grid_x1 = grids.x
            grid_x2 = grids.y
            ff_flag = True

        elif project == 'yx':
            grid_x1 = grids.y
            grid_x2 = grids.x
            ff_flag = True

    comps = list(project)

    if not amp_only:
        fig, ax = pt.subplots(1,2, figsize=(10,5), gridspec_kw={'wspace':0.5})

        if mode == 'linear':
            if norm:
                field_pl = np.absolute(field) / max_field
            else:
                field_pl = np.absolute(field)

            vmin = np.min(field_pl) if vmin is None else vmin
            vmax = np.max(field_pl) if vmax is None else vmax
            
            ampfig = ax[0].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_pl,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], np.angle(field), cmap=cmaps.parula, shading='auto')

        elif mode == 'dB':
            field_dB = 20 * np.log10(np.absolute(field) / max_field)
            
            vmin = np.min(field_dB) if vmin is None else vmin
            vmax = np.max(field_dB) if vmax is None else vmax
            
            if unwrap_phase:
                phase = np.unwrap(np.unwrap(np.angle(field), axis=0), axis=1)

            else:
                phase = np.angle(field)
            
            ampfig = ax[0].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_dB,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], phase, cmap=cmaps.parula, shading='auto')

        divider1 = make_axes_locatable(ax[0])
        divider2 = make_axes_locatable(ax[1])

        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)

        c1 = fig.colorbar(ampfig, cax=cax1, orientation='vertical')
        c2 = fig.colorbar(phasefig, cax=cax2, orientation='vertical')

        if ff_flag:
            ax[0].set_ylabel(r"dEl / {}".format(units[0]))
            ax[0].set_xlabel(r"dAz / {}".format(units[0]))
            ax[1].set_ylabel(r"dEl / {}".format(units[0]))
            ax[1].set_xlabel(r"dAz / {}".format(units[0]))
            ax[0].set_box_aspect(1)
            ax[1].set_box_aspect(1)

        else:
            ax[0].set_ylabel(r"${}$ / {}".format(comps[1], units[0]))
            ax[0].set_xlabel(r"${}$ / {}".format(comps[0], units[0]))
            ax[1].set_ylabel(r"${}$ / {}".format(comps[1], units[0]))
            ax[1].set_xlabel(r"${}$ / {}".format(comps[0], units[0]))

        ax[0].set_title(titleA, y=1.08)
        ax[0].set_aspect(1)
        ax[1].set_title(titleP, y=1.08)
        ax[1].set_aspect(1)


    else:
        fig, ax = pt.subplots(1,1, figsize=(5,5))

        divider = make_axes_locatable(ax)

        cax = divider.append_axes('right', size='5%', pad=0.05)

        if mode == 'linear':
            ampfig = ax.pcolormesh(grid_x1 * units[1], grid_x2 * units[1], np.absolute(field),
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')

        elif mode == 'dB':
            ampfig = ax.pcolormesh(grid_x1 * units[1], grid_x2 * units[1], 20 * np.log10(np.absolute(field) / max_field),
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')

        if ff_flag:
            ax.set_ylabel(r"dEl / {}".format(units[0]))
            ax.set_xlabel(r"dAz / {}".format(units[0]))

        else:
            ax.set_ylabel(r"${}$ / {}".format(comps[1], units[0]))
            ax.set_xlabel(r"${}$ / {}".format(comps[0], units[0]))

        ax.set_title(titleA, y=1.08)
        ax.set_box_aspect(1)

        c = fig.colorbar(ampfig, cax=cax, orientation='vertical')
    
    if aperDict["plot"]:
        xc = aperDict["center"][0]
        yc = aperDict["center"][1]
        Ro = 2*aperDict["outer"]
        Ri = 2*aperDict["inner"]


        try:
            for axx in ax:
                circleo=mpl.patches.Ellipse((xc,yc),Ro[0], Ro[1], color='black', fill=False)
                circlei=mpl.patches.Ellipse((xc,yc),Ri[0], Ri[1], color='black', fill=False)
                
                axx.add_patch(circleo)
                axx.add_patch(circlei)
                axx.scatter(xc, yc, color='black', marker='x')
        
        except:
            ax.add_patch(circleo)
            ax.add_patch(circlei)
            ax.scatter(xc, yc, color='black', marker='x')

    return fig, ax

##
# Plot a 3D reflector.
#
# @param plotObject A reflDict containing surface on which to plot beam. 
# @param ax Axis to use for plotting.
# @param fine Spacing of normals for plotting.
# @param cmap Colormap of reflector.
# @param norm Plot reflector normals.
# @param foc1 Plot focus 1.
# @param foc2 Plot focus 2.
# @param plotSystem_f Whether or not plot3D is called from plotSystem.
def plot3D(plotObject, ax, fine, cmap,
            norm, foc1, foc2, plotSystem_f=False):

    skip = slice(None,None,fine)
    grids = generateGrid(plotObject, transform=True, spheric=True)

    ax.plot_surface(grids.x[skip], grids.y[skip], grids.z[skip],
                   linewidth=0, antialiased=False, alpha=1, cmap=cmap)

    if foc1:
        ax.scatter(plotObject["focus_1"][0], plotObject["focus_1"][1], plotObject["focus_1"][2], color='black')

    if foc2:
        ax.scatter(plotObject["focus_2"][0], plotObject["focus_2"][1], plotObject["focus_2"][2], color='black')

    if norm:
        length = 10# np.sqrt(np.dot(plotObject["focus_1"], plotObject["focus_1"])) / 5
        skipn = slice(None,None,10*fine)
        ax.quiver(grids.x[skipn,skipn], grids.y[skipn,skipn], grids.z[skipn,skipn],
                        grids.nx[skipn,skipn], grids.ny[skipn,skipn], grids.nz[skipn,skipn],
                        color='black', length=length, normalize=True)

    if not plotSystem_f:
        ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
        ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
        ax.set_zlabel(r"$z$ / [mm]", labelpad=50)
        ax.set_title(plotObject["name"], fontsize=20)
        world_limits = ax.get_w_lims()
        ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
        ax.tick_params(axis='x', which='major', pad=-3)
        ax.minorticks_off()

    del grids

##
# Plot the system.
#
# @param systemDict Dictionary containing the reflectors to be plotted.
# @param ax Axis of plot.
# @param fine Spacing of normals for plotting.
# @param cmap Colormap of reflector.
# @param norm Plot reflector normals.
# @param foc1 Plot focus 1.
# @param foc2 Plot focus 2.
# @param RTframes List containing frames to be plotted.
def plotSystem(systemDict, ax, fine, cmap,norm,
            foc1, foc2, RTframes):

    for i, (key, refl) in enumerate(systemDict.items()):
        if isinstance(cmap, list):
            _cmap = cmap[i]

        else:
            _cmap = cmap

        plot3D(refl, ax, fine=fine, cmap=_cmap,
                    norm=norm, foc1=foc1, foc2=foc2, plotSystem_f=True)
    
    ax.set_ylabel(r"$y$ / [mm]", labelpad=20)
    ax.set_xlabel(r"$x$ / [mm]", labelpad=10)
    ax.set_zlabel(r"$z$ / [mm]", labelpad=20)
    #ax.set_title("System", fontsize=20)
    world_limits = ax.get_w_lims()

    ax.set_box_aspect((1,1,1))
    ax.tick_params(axis='x', which='major', pad=-3)

    if RTframes:
        for i in range(RTframes[0].size):
            x = []
            y = []
            z = []

            for frame in RTframes:
                x.append(frame.x[i])
                y.append(frame.y[i])
                z.append(frame.z[i])

            ax.plot(x, y, z, color='grey', zorder=100)


    #set_axes_equal(ax)
    ax.minorticks_off()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

##
# EXPERIMENTAL
def beamCut(plotObject, field, cross='', units='', vmin=-50, vmax=0, frac=1, show=True, save=False, ret=False):

    x_center = int((plotObject["gridsize"][0] - 1) / 2)
    y_center = int((plotObject["gridsize"][1] - 1) / 2)

    grids = generateGrid(plotObject, spheric=False)

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

##
# Plot a ray-trace frame spot diagram.
#
# @param frame A PyPO frame object.
# @param project Set abscissa and ordinate of plot. Should be given as a string.
# @param savePath Path to save plot to.
# @param returns Whether to return figure object.
# @param aspect Aspect ratio of plot.
def plotRTframe(frame, project, savePath, returns, aspect):
    fig, ax = pt.subplots(1,1, figsize=(5,5))

    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)

    if project == "xy":
        ax.scatter(frame.x[idx_good], frame.y[idx_good], color="black", s=10)
        ax.set_xlabel(r"$x$ / mm")
        ax.set_ylabel(r"$y$ / mm")

    elif project == "xz":
        ax.scatter(frame.x[idx_good], frame.z[idx_good], color="black", s=10)
        ax.set_xlabel(r"$x$ / mm")
        ax.set_ylabel(r"$z$ / mm")
    
    elif project == "yz":
        ax.scatter(frame.y[idx_good], frame.z[idx_good], color="black", s=10)
        ax.set_xlabel(r"$y$ / mm")
        ax.set_ylabel(r"$z$ / mm")
    
    elif project == "yx":
        ax.scatter(frame.y[idx_good], frame.x[idx_good], color="black", s=10)
        ax.set_xlabel(r"$y$ / mm")
        ax.set_ylabel(r"$x$ / mm")

    elif project == "zy":
        ax.scatter(frame.z[idx_good], frame.y[idx_good], color="black", s=10)
        ax.set_xlabel(r"$z$ / mm")
        ax.set_ylabel(r"$y$ / mm")
    
    elif project == "zx":
        ax.scatter(frame.z[idx_good], frame.x[idx_good], color="black", s=10)
        ax.set_xlabel(r"$z$ / mm")
        ax.set_ylabel(r"$x$ / mm")

    ax.set_aspect(aspect)
    
    if returns:
        return fig

    pt.show()
