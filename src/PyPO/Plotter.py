import numpy as np
import matplotlib.pyplot as pt
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

import PyPO.PlotConfig
import PyPO.Colormaps as cmaps
from PyPO.BindRefl import *

##
# @file
# File containing functions for generating plots.

##
# Generate a 2D plot of a field or current.
#
# @param plotObject A reflDict containing surface on which to plot beam. 
# @param field PyPO field or current component to plot.
# @param contour A PyPO field or current component to plot as contour.
# @param vmin Minimum amplitude value to display. Default is -30.
# @param vmax Maximum amplitude value to display. Default is 0.
# @param levels Levels for contourplot.
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
def plotBeam2D(plotObject, field, contour,
                vmin, vmax, levels, show, amp_only,
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
                if contour is not None:
                    contour = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour = np.absolute(contour)

            vmin = np.min(field_pl) if vmin is None else vmin
            vmax = np.max(field_pl) if vmax is None else vmax
            
            ampfig = ax[0].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_pl**2,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], np.angle(field), cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont0 = ax[0].contour(grid_x1 * units[1], grid_x2 * units[1], contour**2, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1 * units[1], grid_x2 * units[1], np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)

                ax[0].clabel(cont0)
                ax[1].clabel(cont1)

        elif mode == 'dB':
            if titleA == "Power":
                titleA += " / dB"
            if titleP == "Phase":
                titleP += " / rad"
            field_dB = 20 * np.log10(np.absolute(field) / max_field)
            
            if contour is not None:
                contour_dB = 20 * np.log10(np.absolute(contour) / np.max(np.absolute(contour)))
            
            vmin = np.min(field_dB) if vmin is None else vmin
            vmax = np.max(field_dB) if vmax is None else vmax
            
            if unwrap_phase:
                phase = np.unwrap(np.unwrap(np.angle(field), axis=0), axis=1)

            else:
                phase = np.angle(field)
            
            ampfig = ax[0].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_dB,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            phasefig = ax[1].pcolormesh(grid_x1 * units[1], grid_x2 * units[1], phase, cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont0 = ax[0].contour(grid_x1 * units[1], grid_x2 * units[1], contour_dB, levels, cmap=cm.binary, linewidths=0.5)
                cont1 = ax[1].contour(grid_x1 * units[1], grid_x2 * units[1], np.angle(contour), levels, cmap=cm.binary, linewidths=0.5)
                
                ax[0].clabel(cont0)
                ax[1].clabel(cont1)
        
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
            if norm:
                field_pl = np.absolute(field) / max_field
                if contour is not None:
                    contour_pl = np.absolute(contour) / np.max(np.absolute(contour))
            
            else:
                field_pl = np.absolute(field)
                if contour is not None:
                    contour_pl = np.absolute(contour)

            vmin = np.min(field_pl) if vmin is None else vmin
            vmax = np.max(field_pl) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_pl**2,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')

            if contour is not None:
                cont = ax.contour(grid_x1 * units[1], grid_x2 * units[1], contour_pl**2, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)
        
        elif mode == 'dB':
            if titleA == "Power":
                titleA += " / dB"
            field_dB = 20 * np.log10(np.absolute(field) / max_field)
            
            if contour is not None:
                contour_dB = 20 * np.log10(np.absolute(contour) / np.max(np.absolute(contour)))
            
            vmin = np.min(field_dB) if vmin is None else vmin
            vmax = np.max(field_dB) if vmax is None else vmax
            
            ampfig = ax.pcolormesh(grid_x1 * units[1], grid_x2 * units[1], field_dB,
                                    vmin=vmin, vmax=vmax, cmap=cmaps.parula, shading='auto')
            
            if contour is not None:
                cont = ax.contour(grid_x1 * units[1], grid_x2 * units[1], contour_dB, levels, cmap=cm.binary, linewidths=0.5)
                ax.clabel(cont)

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
            foc1, foc2, RTframes, RTcolor):

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

            ax.plot(x, y, z, color=RTcolor, zorder=100, lw=0.7)


    #set_axes_equal(ax)
    ax.minorticks_off()
    #ax.set_box_aspect((1,1,1))
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

##
# Plot two beam cuts in the same figure.
#
# @param x_cut E-plane.
# @param y_cut H-plane.
# @param x_strip Co-ordinates for plotting E-plane.
# @param y_strip Co-ordinates for plotting H-plane.
# @param vmin Minimum for plot range.
# @param vmax Maximum for plot range.
# @param unit for x-axis.
#
# @returns fig Plot figure.
# @returns ax Plot axis.
def plotBeamCut(x_cut, y_cut, x_strip, y_strip, vmin, vmax, unit):
    fig, ax = pt.subplots(1,1, figsize=(5,5))

    ax.plot(x_strip * unit[1], x_cut, color="blue", label="E-plane")
    ax.plot(y_strip * unit[1], y_cut, color="red", ls="dashed", label="H-plane")

    ax.set_xlim(np.min(x_strip * unit[1]), np.max(x_strip * unit[1]))
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel(r"$\theta$ / {}".format(unit[0]))
    ax.set_ylabel("Power / dB")
    ax.legend(frameon=False, prop={'size': 13},handlelength=1)

    return fig, ax


##
# Plot a ray-trace frame spot diagram.
#
# @param frame A PyPO frame object.
# @param project Set abscissa and ordinate of plot. Should be given as a string.
# @param savePath Path to save plot to.
# @param returns Whether to return figure object.
# @param aspect Aspect ratio of plot.
# @param unit Units of the axes for the plot.
def plotRTframe(frame, project, savePath, returns, aspect, unit):
    fig, ax = pt.subplots(1,1, figsize=(5,5))

    idx_good = np.argwhere((frame.dx**2 + frame.dy**2 + frame.dz**2) > 0.8)

    if project == "xy":
        ax.scatter(frame.x[idx_good] * unit[1], frame.y[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$x$ / {}".format(unit[0]))
        ax.set_ylabel(r"$y$ / {}".format(unit[0]))

    elif project == "xz":
        ax.scatter(frame.x[idx_good] * unit[1], frame.z[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$x$ / {}".format(unit[0]))
        ax.set_ylabel(r"$z$ / {}".format(unit[0]))
    
    elif project == "yz":
        ax.scatter(frame.y[idx_good] * unit[1], frame.z[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$y$ / {}".format(unit[0]))
        ax.set_ylabel(r"$z$ / {}".format(unit[0]))
    
    elif project == "yx":
        ax.scatter(frame.y[idx_good] * unit[1], frame.x[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$y$ / {}".format(unit[0]))
        ax.set_ylabel(r"$x$ / {}".format(unit[0]))

    elif project == "zy":
        ax.scatter(frame.z[idx_good] * unit[1], frame.y[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$z$ / {}".format(unit[0]))
        ax.set_ylabel(r"$y$ / {}".format(unit[0]))
    
    elif project == "zx":
        ax.scatter(frame.z[idx_good] * unit[1], frame.x[idx_good] * unit[1], color="black", s=10)
        ax.set_xlabel(r"$z$ / {}".format(unit[0]))
        ax.set_ylabel(r"$x$ / {}".format(unit[0]))

    ax.set_aspect(aspect)
    
    if returns:
        return fig

    pt.show()
