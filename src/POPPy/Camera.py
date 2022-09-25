import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp

import src.POPPy.MatRotate as MatRotate

class Camera(object):
    """
    Class for instantiating a camera object. If a planar reflector is created, actually a camera object is instantiated.
    NOTE: if prop_mode for PO == 1, it is assumed camera is in FAR-FIELD. x and y then correspond to Az-El.
    Center only affects pointing then!
    """
    
    #### DUNDER METHODS ###
    
    def __init__(self, center, name, units):
        if isinstance(units, list):
            self.units = units[0]
            self.unit_spat = units[1]
            self.conv_spat = self._get_conv(self.unit_spat)
            
            self.center = center * self.conv_spat
            self.cRot = center
            self.conv = self._get_conv(self.units)
            
        else:
            self.conv = self._get_conv(units)
            self.center = center * self.conv
            self.cRot = center

        self.elType = "Camera"
        self.name = name
        
        # Use internal list of references to iterable attributes
        self._iterList = [0 for _ in range(7)]
        
        self.norm = np.array([0,0,1])
        
        self.ff_flag = False
        
    def __str__(self):
        s = """\n######################### CAMERA INFO #########################
Center position     : [{:.3f}, {:.3f}, {:.3f}] [mm]
######################### CAMERA INFO #########################\n""".format(self.center[0], self.center[1], self.center[2])
        return s
    
    def __iter__(self):
        self._iterIdx = 0
        return self
    
    def __next__(self):
        if self._iterIdx < len(self._iterList):
            result = self._iterList[self._iterIdx]
            self._iterIdx += 1
            
            return result
        
        else:
            raise StopIteration
        
    #### PUBLIC METHODS ###
        
    def setGrid(self, lims_x, lims_y, gridsize, gmode):
        """
        (PUBLIC)
        Set the xyz, area and normal vector grids to camera surface.
        
        @param  ->
            lims_x      :   Lower and upper limits on x (u) values
            lims_y      :   Lower and upper limits on y (v) values
            gridsize    :   Number of cells along the xy (uv) axes
            gmode       :   Use direct xy function evaluation, a uv parametrization or Az over El co-ordinates
        """
        
        self.shape = gridsize
        
        if gmode == 'xy':
            grid_x, grid_y = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            grid_x *= self.conv
            grid_y *= self.conv
        
            dx = grid_x[1,0] - grid_x[0,0]
            dy = grid_y[0,1] - grid_y[0,0]
            
            self.area = np.ones(grid_x.shape) * dx * dy
            
        elif gmode == 'uv':
            lims_x[0] *= self.conv
            lims_x[1] *= self.conv
            
            grid_u, grid_v = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            grid_x = grid_u * np.cos(grid_v)
            grid_y = grid_u * np.sin(grid_v)
            
            du = grid_u[1,0] - grid_u[0,0]
            dv = grid_v[0,1] - grid_v[0,0]
            
            self.area = grid_u * du * dv
            
        elif gmode == 'AoE':
            # Since only used for far-field, immediately convert to spherical coords
            grid_Az, grid_El = np.mgrid[lims_x[0]:lims_x[1]:gridsize[0]*1j, lims_y[0]:lims_y[1]:gridsize[1]*1j]
            
            # Convert first to radians
            grid_Az *= self.conv
            grid_El *= self.conv
            
            # Store Az and El for plotting purps
            self.grid_Az = grid_Az
            self.grid_El = grid_El
            
            dAz = grid_Az[1,0] - grid_Az[0,0]
            dEl = grid_El[0,1] - grid_El[0,0]
            
            # theta
            grid_x = np.sqrt(grid_Az**2 + grid_El**2)
            
            # phi
            grid_y = np.arctan(grid_El / grid_Az)

            toFill = np.argwhere(np.isnan(grid_y))
            
            
            
            if isinstance(toFill, np.ndarray):
                a, b = toFill[0]

                grid_y[a, b] = 0
            
            pt.imshow(grid_y)
            pt.show()
            
            self.area = dAz * dEl # whatever...
            self.ff_flag = True
        
        self.grid_x = grid_x + self.center[0]
        self.grid_y = grid_y + self.center[1]
        self.grid_z = np.zeros(self.grid_x.shape) + self.center[2]
        
        self.grid_nx = np.zeros(self.grid_x.shape)
        self.grid_ny = np.zeros(self.grid_y.shape)
        self.grid_nz = np.ones(self.grid_z.shape)
        
        
        
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
       
    def set_cRot(self, cRot, units='mm'):
        """
        (PUBLIC)
        Set center of rotation for camera.
        
        @param  ->
            cRot        :   Array containg co-ordinates of center of rotation
            units       :   Units of co-ordinate
        """
        conv = self.get_conv(units)
        self.cRot = cRot * conv
    
    def translateGrid(self, offTrans, units='mm'):
        """
        (PUBLIC)
        Translate camera along xyz axes.
        
        @param  ->
            offTrans    :   Translation in xyz   
            units       :   Units of the supplied translation
        """
        
        conv = self._get_conv(units)
        self.grid_x += offTrans[0] * conv
        self.grid_y += offTrans[1] * conv
        self.grid_z += offTrans[2] * conv
        
        self.center += offTrans
        
        self._updateIterlist()
        
    def rotateGrid(self, offRot, radians=False):
        """
        (PUBLIC)
        Rotate camera around xyz axes.
        
        @param  ->
            offRot      :   Rotation around xyz axes, in degrees
            radians     :   Whether offRot is in degrees or radians
        """
        
        gridRot = MatRotate.MatRotate(offRot, [self.grid_x, self.grid_y, self.grid_z], self.cRot, radians=radians)
        
        self.grid_x = gridRot[0]
        self.grid_y = gridRot[1]
        self.grid_z = gridRot[2]
        
        grid_nRot = MatRotate.MatRotate(offRot, [self.grid_nx, self.grid_ny, self.grid_nz], self.cRot, vecRot=True, radians=radians)
        
        self.grid_nx = grid_nRot[0]
        self.grid_ny = grid_nRot[1]
        self.grid_nz = grid_nRot[2]
        
        self.center = MatRotate.MatRotate(offRot, self.center, self.cRot, radians=radians)
        self.norm = MatRotate.MatRotate(offRot, self.norm, self.cRot, vecRot=True, radians=radians)
        
        self._updateIterlist()
    
    def interpCamera(self, res, mode):
        """
        (PUBLIC)
        Obtain tcks parameters for interpolation on camera.
        
        @param  ->
            res         :   Space between points on surface to use for interpolation.
                        :   If too low, overflow error might occur.
            mode        :   Dependent axis for interpolation
        """
        
        skip = slice(None,None,res)

        if mode == 'z':
            posInterp = interp.bisplrep(self.grid_x, self.grid_y, self.grid_z)
        
            nxInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nx.ravel()[skip])
            nyInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_ny.ravel()[skip])
            nzInterp = interp.bisplrep(self.grid_x.ravel()[skip], self.grid_y.ravel()[skip], self.grid_nz.ravel()[skip])

        elif mode == 'x':
            posInterp = interp.bisplrep(self.grid_y, self.grid_z, self.grid_x)
        
            nxInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_nx.ravel()[skip])
            nyInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_ny.ravel()[skip])
            nzInterp = interp.bisplrep(self.grid_y.ravel()[skip], self.grid_z.ravel()[skip], self.grid_nz.ravel()[skip])
            
        elif mode == 'y':
            posInterp = interp.bisplrep(self.grid_z, self.grid_x, self.grid_y)
        
            nxInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_nx.ravel()[skip])
            nyInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_ny.ravel()[skip])
            nzInterp = interp.bisplrep(self.grid_z.ravel()[skip], self.grid_x.ravel()[skip], self.grid_nz.ravel()[skip])
        
        tcks = [posInterp, nxInterp, nyInterp, nzInterp]
        
        # Store interpolations as members
        self.tcks = tcks
        
    def plotCamera(self, color='gold', returns=False, ax_append=False, norm=False):
        if not ax_append:
            fig, ax = pt.subplots(figsize=(10,10), subplot_kw={"projection": "3d"})
            ax_append = ax

        camera = ax_append.plot_surface(self.grid_x, self.grid_y, self.grid_z,
                       linewidth=0, antialiased=False, alpha=0.5, color=color)

        if norm:
            skipn = slice(None,None,10*fine)
            ax_append.quiver(self.grid_x[skipn], self.grid_y[skipn], self.grid_z[skipn], self.grid_nx[skipn], self.grid_ny[skipn], self.grid_nz[skipn], color='black', length=100, normalize=True)

        if not returns:
            ax_append.set_ylabel(r"$y$ / [mm]", labelpad=20)
            ax_append.set_xlabel(r"$x$ / [mm]", labelpad=10)
            ax_append.set_zlabel(r"$z$ / [mm]", labelpad=50)
            world_limits = ax_append.get_w_lims()
            ax_append.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
            ax_append.tick_params(axis='x', which='major', pad=-3)
            pt.show()
            
        else:
            return ax_append
        
    #### PRIVATE METHODS ###
        
    def _get_conv(self, units):
        """
        (PRIVATE)
        Get conversion factor given the units supplied.
        
        @param  ->
            units       :   Units of the supplied translation
            
        @return ->
            conv        :   Conversion factor for unit to mm conversion
        """
        
        conv = 0
        
        if units == 'mm':
            conv = 1
        elif units == 'cm':
            conv = 1e2
        elif units == 'm':
            conv = 1e3
        
        # Angular units - convert here to radians
        elif units == 'deg':
            conv = np.pi / 180
        elif units == 'am':
            conv = np.pi / (180 * 60)
        elif units == 'as':
            conv = np.pi / (180 * 3600)
        
        
        return conv
    
    def _updateIterlist(self):
        """
        (PRIVATE)
        Update internal iteration container.
        Called after translation and rotation.
        """
        
        self._iterList[0] = self.grid_x
        self._iterList[1] = self.grid_y
        self._iterList[2] = self.grid_z
        
        self._iterList[3] = self.area
        
        self._iterList[4] = self.grid_nx
        self._iterList[5] = self.grid_ny
        self._iterList[6] = self.grid_nz
