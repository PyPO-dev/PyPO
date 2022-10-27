import math
import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as pt
import scipy.interpolate as interp
import scipy.optimize as optimize
from itertools import islice
from multiprocessing import  Pool
from functools import partial

import src.POPPy.MatRotate as MatRotate
import src.POPPy.Copy as Copy

class Frame(object):
    def __init__(self, length):
        self.x = np.zeros(length)
        self.y = np.zeros(length)
        self.z = np.zeros(length)

        self.dx = np.zeros(length)
        self.dy = np.zeros(length)
        self.dz = np.zeros(length)

        self.length = length

    def assignValues(self, x, y, z, dx, dy, dz):
        self.x = x
        self.y = y
        self.z = z

        self.dx = dx
        self.dy = dy
        self.dz = dz

    # Convert self.x, self.y, etc... into lists of arrays.
    # Because this is only applied to _frame objects, can
    # modify itself.
    def chunkify(self, nReduced):
        xl = []
        yl = []
        zl = []

        dxl = []
        dyl = []
        dzl = []

        start = 0

        while start <= self.length:
            stop = start + nReduced

            if stop > self.length:
                stop = self.length

            xl.append(self.x[start:stop])
            yl.append(self.y[start:stop])
            zl.append(self.z[start:stop])

            dxl.append(self.dx[start:stop])
            dyl.append(self.dy[start:stop])
            dzl.append(self.dz[start:stop])

            start += nReduced

        self.x = xl
        self.y = yl
        self.z = zl

        self.dx = dxl
        self.dy = dyl
        self.dz = dzl

    def writoToFrame(self, idx, xyz, dxyz):
        self.x[idx] = xyz[0]
        self.y[idx] = xyz[1]
        self.z[idx] = xyz[2]

        self.dx[idx] = dxyz[0]
        self.dy[idx] = dxyz[1]
        self.dz[idx] = dxyz[2]

class Beam(object):
    def __init__(self, frame, nTot, frameCounter):
        self.frameDict = {"frame0" : frame}

        self.nTot = nTot
        self.frameCounter = frameCounter

    @classmethod
    def initFromInp(cls, nRays, nRing,
                     a, b, angx, angy,
                     originChief, tiltChief):

        """
        (PUBLIC)
        Initialize a ray-trace from beam parameters.
        We initialize a beam object by appending frames. A frame consists of 6 lists of
        length nRays * nRing * 4 + 1. Each ray trace, a new frame is calculated.
        One can choose to save previous frames, or keep only the new one.
        The frames are stored in frameDict. When one chooses to remove a frame, the frameDict
        will be emptied to store memory. The entry is deleted. Frame numbering continues though,
        so the keys stay the same.

        @param  ->
            nRays           :   Number of rays in a concentric ring
            nRing           :   Number of concentric ray-trace rings
            a               :   Semi major axis of ring, in mm
            b               :   Semi minor axis of ring, in mm
            angx            :   Opening angle of beam along x-axis, degrees
            angy            :   Opening angle of beam along y-axis, degrees
            originChief     :   Array containg co-ordinates of chief ray origin
            tiltChief       :   Tilt of chief ray w.r.t. normal vector along z axis
        """

        nomChief = np.array([0,0,1]) # Always initialize raytrace beam along z-axis

        originChief = originChief
        directionChief = MatRotate.MatRotate(tiltChief, nomChief, vecRot=True)

        nTot = 1 + nRing * 4 * nRays
        frameCounter = 0

        frame0 = Frame(nTot)

        alpha = 0.0                             # Set first circle ray in the right of beam
        if nRays > 0:
            d_alpha = 2 * np.pi / (4 * nRays)   # Set spacing in clockwise angle
        else:
            d_alpha = 0

        n = 1
        for i in range(nTot):

            if i == 0: # Chief ray
                frame0.writoToFrame(i, originChief, directionChief)
                continue

            pos_ray = np.array([a * np.cos(alpha), b * np.sin(alpha), 0]) / nRing * n
            rotation = np.array([np.radians(angy) * np.sin(alpha) / nRing * n, np.radians(angx) * np.cos(alpha) / nRing * n, 2*alpha])

            direction = MatRotate.MatRotate(rotation, nomChief, vecRot=True, radians=True)
            direction = MatRotate.MatRotate(tiltChief, direction, vecRot=True)

            pos_r = MatRotate.MatRotate(tiltChief, originChief + pos_ray, origin=originChief)

            frame0.writoToFrame(i, pos_r, direction)
            alpha += d_alpha

            if i == int(nTot / nRing) * n:
                n += 1
                alpha = 0

        return cls(frame0, nTot, frameCounter)

    @classmethod
    def initFromPO(cls, source, Pr):
        """
        (PUBLIC)
        Initialize a ray-trace from PO calculation.
        @param  ->
        source          :   Object on which Pr is calculated
        Pr              :   List containing components of Pr
        """

        x = source.grid_x.flatten()
        y = source.grid_y.flatten()
        z = source.grid_z.flaten()

        dx = Pr[0].flatten()
        dy = Pr[1].flatten()
        dz = Pr[2].flatten()

        nTot = len(x)
        frameCounter = 0
        frame0 = Frame(nTot)

        frame0.assignValues(x, y, z, dx, dy, dz)

        return cls(frame0, nTot, frameCounter)

    # Iterate over frames in beam
    def __iter__(self):
        yield from self.frameDict.items()

    def appendFrame(self, frame):
        self.frameDict["frame{}".format(self.frameCounter)] = frame

    def _convertToArr(self, frame, idx):
        xyz = np.zeros(3)
        dxyz = np.zeros(3)

        xyz[0] = frame.x[idx]
        xyz[1] = frame.y[idx]
        xyz[2] = frame.z[idx]

        dxyz[0] = frame.dx[idx]
        dxyz[1] = frame.dy[idx]
        dxyz[2] = frame.dz[idx]

        return xyz, dxyz



class RayTrace(object):
    def initRaytracer(self, nRays, nRing,
                 a, b, angx, angy,
                 originChief, tiltChief):
        """
        (PUBLIC)
        Initialize a ray-trace from beam parameters.
        See Beam class for explanation of arguments.
        """
        self.beam = Beam.initFromInp(nRays, nRing,
                     a, b, angx, angy,
                     originChief, tiltChief)

    def POtoRaytracer(self, source, Pr):
        """
        (PUBLIC)
        Initialize a ray-trace from PO calculation.

        @param  ->
            source          :   Object on which Pr is calculated
            Pr              :   List containing components of Pr
        """
        self.beam = Beam.initFromPO(source, Pr)

    def sizeOf(self, units='b'):
        """
        (PUBLIC)
        Return memory size of self.beam.

        @param  ->
            units           :   Units of memory size for return.
                            :   Options are bytes 'b', megabytes 'mb' and gigabytes 'gb'

        @return ->
            size            :   Memory size of self.rays object
        """

        factor = 1

        if units == 'kb':
            factor = 1e-3

        elif units == 'mb':
            factor = 1e-6

        elif units == 'gb':
            factor = 1e-9

        # Size of references to frames is beam
        size = sys.getsizeof(self.beam)

        for key, frame in self.beam:
            # Add size of reference to frame
            size += sys.getsizeof(frame)

            size += sys.getsizeof(frame.x)
            size += sys.getsizeof(frame.y)
            size += sys.getsizeof(frame.z)

            size += sys.getsizeof(frame.dx)
            size += sys.getsizeof(frame.dy)
            size += sys.getsizeof(frame.dz)

        return size * factor

    def propagateRays(self, target, a0, mode, workers=1):
        """
        (PUBLIC)
        Propagate beam of rays from last frame to new frame on target.

        @param  ->
            target          :   Target surface for propagation
            a0              :   Initial value for optimizing ray to target surface
            mode            :   Dependent variable of interpolation
            workers         :   Number of workers to spawn on CPU
        """

        _frame = Frame(self.beam.nTot)
        _frame_out = Frame(self.beam.nTot)

        self.nReduced = math.ceil(self.beam.nTot / workers)

        mat, mat_v = target.getFullTransform()
        inv_mat, inv_mat_v = target.getFullTransform(fwd=False)

        self._transformRays(self.beam.frameDict["frame{}".format(self.beam.frameCounter)], _frame, inv_mat, inv_mat_v)
        #_frame = self.beam.frameDict["frame{}".format(self.beam.frameCounter)]
        # Evaluate _propagateRays partially
        #self.beam.frameDict["frame0"] = Copy.copyGrid(_frame)

        _propagateRayspar = partial(self._propagateRays, target=target, a0=a0)

        PIDs = np.arange(workers)

        # Pack _frame into chunks of rays
        _frame.chunkify(self.nReduced)

        args = zip(_frame.x, _frame.y, _frame.z,
                    _frame.dx, _frame.dy, _frame.dz, PIDs)

        pool = Pool(workers)

        out = pool.map(_propagateRayspar, args)
        pool.close()


        _frame_out = self._reconstructToFrame(out)
        __frame = Frame(self.beam.nTot)

        # Delete chunkified frame, create new empty one
        del _frame

        self.beam.frameCounter += 1
        self._transformRays(_frame_out, __frame, mat, mat_v)
        #__frame = _frame_out

        pt.plot(__frame.z)
        pt.show()


        self.beam.appendFrame(__frame)

        del _frame_out

    def clearBeam(self):
        """
        (PUBLIC)
        Clear current self.rays object
        """
        self.beam.clear()

    def clearFrame(self, idx):
        del self.beam.frameDict["frame{}".format(idx)]

    def plotRays(self, quiv=True, frame=0, mode='z', save=False):
        fig, ax = pt.subplots(1,1)

        toPlot = self.beam.frameDict["frame{}".format(frame)]

        #print(self.beam.frameDict["frame{}".format(frame)].x)

        if mode == 'z':
            ax.set_xlabel('$x$ / [mm]')
            ax.set_ylabel('$y$ / [mm]')
            ax.scatter(toPlot.x, toPlot.y, color='black', s=10)

            if quiv:
                ax.quiver(ray["positions"][frame][0], ray["positions"][frame][1], 10 * ray["directions"][frame][0], 10 * ray["directions"][frame][1], color='black', width=0.005, scale=10)

        elif mode == 'x':
            ax.set_xlabel('$y$ / [mm]')
            ax.set_ylabel('$z$ / [mm]')
            ax.scatter(toPlot.y, toPlot.z, color='black', s=10)

            if quiv:
                ax.quiver(ray["positions"][frame][1], ray["positions"][frame][2], 10 * ray["directions"][frame][1], 10 * ray["directions"][frame][2], color='black', width=0.005, scale=10)

        elif mode == 'y':
            ax.set_xlabel('$z$ / [mm]')
            ax.set_ylabel('$x$ / [mm]')
            ax.scatter(toPlot.z, toPlot.x, color='black', s=10)

            if quiv:
                ax.quiver(ray["positions"][frame][2], ray["positions"][frame][0], 10 * ray["directions"][frame][2], 10 * ray["directions"][frame][0], color='black', width=0.005, scale=10)

        ax.set_aspect(1)
        ax.set_title('Spot diagram, frame = {}'.format(frame))

        if save:
            pt.savefig(fname='spot_{}.jpg'.format(frame),bbox_inches='tight', dpi=300)

        pt.show()

    def plotRaysSystem(self, ax_append):

        for i in range(self.beam.nTot):
            x = []
            y = []
            z = []
            for key, frame in self.beam:

                x.append(frame.x[i])
                y.append(frame.y[i])
                z.append(frame.z[i])

            ax_append.plot(x, y, z, color='grey')

        return ax_append

    #### PRIVATE METHODS ###

    def _propagateRays(self, args, target, a0):
        """
        (PRIVATE)
        Calculate position and reflected normal vector of ray on target surface.

        @param  ->
            args            :   Tuple containing rays and process IDs for this worker
            a0              :   Initial value for ray-tracer
        """

        x, y, z, dx, dy, dz, PID = args

        j = 0 # Counter index

        for i in range(len(x)):
            #print(dy[i])
            getxyz = lambda a : target.xyGrid(a*dx[i] + x[i], a*dy[i] + y[i])
            #diff = lambda a : np.sqrt((getxyz(a)[0] - a*dx[i] - x[i])**2 + (getxyz(a)[1] - a*dy[i] - y[i])**2 + (getxyz(a)[2] - a*dz[i] + z[i])**2)
            diff = lambda a : np.absolute(getxyz(a)[2] - a*dz[i] + z[i])
            #print(diff(z[i]))

            a_opt = optimize.fmin(diff, a0, disp=0, xtol=1e-16, ftol=1e-16)[0]

            xo, yo, zo, nx, ny, nz = target.xyGrid(a_opt*dx[i] + x[i], a_opt*dy[i] + y[i])

            norm = np.sqrt(nx**2 + ny**2 + nz**2)

            nx /= norm
            ny /= norm
            nz /= norm

            x[i] = xo
            y[i] = yo
            z[i] = zo

            dinn = dx[i]*nx + dy[i]*ny + dz[i]*nz

            dx[i] = dx[i] - 2 * dinn * nx
            dy[i] = dy[i] - 2 * dinn * ny
            dz[i] = dz[i] - 2 * dinn * nz

            #print(a_opt)

            if (i/self.nReduced * 100) > j and PID == 0:
                print("{} / 100".format(j), end='\r')
                j += 1

        return [x, y, z, dx, dy, dz]

    def _transformRays(self, frame, frame_out, mat, mat_v):
        _vec = np.zeros(4)

        print(mat)
        print(mat_v)

        for i in range(self.beam.nTot):
            _vec[0] = frame.x[i]
            _vec[1] = frame.y[i]
            _vec[2] = frame.z[i]
            _vec[3] = 1

            _vec_out = np.matmul(mat, _vec)

            frame_out.x[i] = _vec_out[0]
            frame_out.y[i] = _vec_out[1]
            frame_out.z[i] = _vec_out[2]

            _vec[0] = frame.dx[i]
            _vec[1] = frame.dy[i]
            _vec[2] = frame.dz[i]
            _vec[3] = 1

            _vec_out = np.matmul(mat_v, _vec)

            frame_out.dx[i] = _vec_out[0]
            frame_out.dy[i] = _vec_out[1]
            frame_out.dz[i] = _vec_out[2]

    def _reconstructToFrame(self, out):
        frame_out = Frame(self.beam.nTot)

        offset = 0
        for subl in out:
            lensubl = len(subl[0]) + offset

            start = offset

            frame_out.x[start:lensubl] = subl[0]
            frame_out.y[start:lensubl] = subl[1]
            frame_out.z[start:lensubl] = subl[2]

            frame_out.dx[start:lensubl] = subl[3]
            frame_out.dy[start:lensubl] = subl[4]
            frame_out.dz[start:lensubl] = subl[5]

            offset = lensubl

        return frame_out


if __name__ == "__main__":
    print("Raytracer class.")
