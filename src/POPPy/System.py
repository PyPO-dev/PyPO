# Standard Python imports
import numbers
import numpy as np
import matplotlib.pyplot as pt
import time
import psutil
import os
import sys
import json

# POPPy-specific modules
from src.POPPy.BindRefl import *
from src.POPPy.BindGPU import *
from src.POPPy.BindCPU import *
from src.POPPy.Copy import copyGrid
from src.POPPy.MatTransform import *
from src.POPPy.Plotter import Plotter
from src.POPPy.POPPyTypes import *

class System(object):
    customBeamPath = './custom/beam/'
    customReflPath = './custom/reflector/'

    savePathElem = './save/elements/'

    def __init__(self):
        self.num_ref = 0
        self.num_cam = 0
        self.system = {}

    def __str__(self):
        pass

    def setCustomBeamPath(self, path, append=False):
        if append:
            self.customBeamPath += path
        else:
            self.customBeamPath = path

    def setCustomReflPath(self, path, append=False):
        if append:
            self.customReflPath += path
        else:
            self.customReflPath = path

    def addPlotter(self, save='./images/'):
        self.plotter = Plotter(save='./images/')

    #### ADD REFLECTOR METHODS
    # Parabola takes as input the reflectordict from now on.
    # Should build correctness test!
    #def addParabola(self, coef, lims_x, lims_y, gridsize, cRot=np.zeros(3), pmode='man', gmode='xy', name="Parabola", axis='a', units='mm', trunc=False, flip=False):
    def addParabola(self, reflDict):
        """
        Function for adding paraboloid reflector to system. If gmode='uv', lims_x should contain the smallest and largest radius and lims_y
        should contain rotation.
        """
        if reflDict["name"] == "Parabola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 0
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

            ve = reflDict["vertex"] # Vertex point position
            f1 = reflDict["focus_1"] # Focal point position

            diff = f1 - ve

            df = np.sqrt(np.dot(diff, diff))
            a = 2 * np.sqrt(df)
            b = a

            orientation = diff / np.sqrt(np.dot(diff, diff))
            offTrans = ve

            # Find rotation in frame of vertex
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0

            offRot = np.array([rx, ry, rz])
            cRot = offTrans

            self.system[reflDict["name"]]["transf"] = MatRotate(offRot, reflDict["transf"], cRot)
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, reflDict["transf"])

            self.system[reflDict["name"]]["coeffs"][0] = a
            self.system[reflDict["name"]]["coeffs"][1] = b
            self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            # Convert v in degrees to radians
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addHyperbola(self, reflDict):
        if reflDict["name"] == "Hyperbola":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 1
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            self.system[reflDict["name"]]["coeffs"] = np.zeros(3)
            # Calculate a, b, c of hyperbola
            f1 = reflDict["focus_1"] # Focal point 1 position
            f2 = reflDict["focus_2"] # Focal point 2 position
            ecc = reflDict["ecc"] # Eccentricity of hyperbola

            diff = f1 - f2
            c = np.sqrt(np.dot(diff, diff)) / 2
            a = c / ecc
            b = np.sqrt(c**2 - a**2)

            # Convert 2D hyperbola a,b,c to 3D hyperboloid a,b,c
            a3 = b
            b3 = b
            c3 = a

            # Find direction between focii
            orientation = diff / np.sqrt(np.dot(diff, diff))

            # Find offset from center. Use offset as rotation origin for simplicity
            center = (f1 + f2) / 2
            offTrans = center

            # Find rotation in frame of center
            rx = np.arccos(1 - np.dot(np.array([1,0,0]), orientation))
            ry = np.arccos(1 - np.dot(np.array([0,1,0]), orientation))
            rz = 0

            offRot = np.array([rx, ry, rz])
            cRot = offTrans

            self.system[reflDict["name"]]["transf"] = MatRotate(offRot, reflDict["transf"], cRot)
            self.system[reflDict["name"]]["transf"] = MatTranslate(offTrans, reflDict["transf"])

            self.system[reflDict["name"]]["coeffs"][0] = a3
            self.system[reflDict["name"]]["coeffs"][1] = b3
            self.system[reflDict["name"]]["coeffs"][2] = c3

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addEllipse(self, reflDict):
        if reflDict["name"] == "Ellipse":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 2
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)

        if reflDict["pmode"] == "focus":
            pass

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["lims_v"] = [self.system[reflDict["name"]]["lims_v"][0],
                                                        self.system[reflDict["name"]]["lims_v"][1]]

            self.system[reflDict["name"]]["gmode"] = 1

        self.num_ref += 1

    def addPlane(self, reflDict):
        if reflDict["name"] == "Plane":
            reflDict["name"] = reflDict["name"] + "_{}".format(self.num_ref)

        reflDict["type"] = 3
        reflDict["transf"] = np.eye(4)

        self.system[reflDict["name"]] = copyGrid(reflDict)
        self.system[reflDict["name"]]["coeffs"] = np.zeros(3)

        self.system[reflDict["name"]]["coeffs"][0] = -1
        self.system[reflDict["name"]]["coeffs"][1] = -1
        self.system[reflDict["name"]]["coeffs"][2] = -1

        if reflDict["gmode"] == "xy":
            self.system[reflDict["name"]]["gmode"] = 0

        elif reflDict["gmode"] == "uv":
            self.system[reflDict["name"]]["gmode"] = 1

        elif reflDict["gmode"] == "AoE":
            # Assume is given in degrees
            # Convert Az and El to radians

            self.system[reflDict["name"]]["lims_Az"] = [self.system[reflDict["name"]]["lims_Az"][0],
                                                        self.system[reflDict["name"]]["lims_Az"][1]]

            self.system[reflDict["name"]]["lims_El"] = [self.system[reflDict["name"]]["lims_El"][0],
                                                        self.system[reflDict["name"]]["lims_El"][1]]

            self.system[reflDict["name"]]["gmode"] = 2

        self.num_ref += 1

    def rotateGrids(self, name, rotation, cRot=np.zeros(3)):
        self.system[name]["transf"] = MatRotate(rotation, self.system[name]["transf"], cRot)

    def translateGrids(self, name, translation):
        self.system[name]["transf"] = MatTranslate(translation, self.system[name]["transf"])

    def generateGrids(self, name):
        grids = generateGrid(self.system[name])
        return grids

    def readCustomBeam(self, name, comp, shape, convert_to_current=True, mode="PMC", ret="currents"):
        rfield = np.loadtxt(self.customBeamPath + "r" + name + ".txt")
        ifield = np.loadtxt(self.customBeamPath + "i" + name + ".txt")

        field = rfield.reshape(shape) + 1j*ifield.reshape(shape)

        fields_c = self._compToFields(comp, field)

        if convert_to_current:

            comps_M = np.zeros((shape[0], shape[1], 3), dtype=complex)
            np.stack((fields_c.Ex, fields_c.Ey, fields_c.Ez), axis=2, out=comps_M)

            comps_J = np.zeros((shape[0], shape[1], 3), dtype=complex)
            np.stack((fields_c.Hx, fields_c.Hy, fields_c.Hz), axis=2, out=comps_J)

            null = np.zeros(field.shape, dtype=complex)

            # Because we initialize in xy plane, take norm as z
            norm = np.array([0,0,1])
            if mode == 'None':
                M = -np.cross(norm, comps_M, axisb=2)
                J = np.cross(norm, comps_J, axisb=2)

                Mx = M[:,:,0]
                My = M[:,:,1]
                Mz = M[:,:,2]

                Jx = J[:,:,0]
                Jy = J[:,:,1]
                Jz = J[:,:,2]

                currents_c = currents(Jx, Jy, Jz, Mx, My, Mz)

            elif mode == 'PMC':
                M = -2 * np.cross(norm, comps_M, axisb=2)

                Mx = M[:,:,0]
                My = M[:,:,1]
                Mz = M[:,:,2]

                currents_c = currents(null, null, null, Mx, My, Mz)

            elif mode == 'PEC':
                J = 2 * np.cross(norm, comps_J, axisb=2)

                Jx = J[:,:,0]
                Jy = J[:,:,1]
                Jz = J[:,:,2]

                currents_c = currents(Jx, Jy, Jz, null, null, null)

        if ret == "currents":
            return currents_c

        elif ret == "fields":
            return fields_c

        elif ret == "both":
            return currents_c, fields_c

    def propagatePO_CPU(self, source_name, target_name, s_currents, k,
                    epsilon=1, t_direction=-1, nThreads=1,
                    mode="JM", precision="double"):

        source = self.system[source_name]
        target = self.system[target_name]

        if precision == "double":
            out = POPPy_CPUd(source, target, s_currents, k, epsilon, t_direction, nThreads, mode)

        return out

    def propagatePO_GPU(self, source_name, target_name, s_currents, k,
                    epsilon=1, t_direction=-1, nThreads=256,
                    mode="JM", precision="single"):

        source = self.system[source_name]
        target = self.system[target_name]

        if precision == "single":
            out = POPPy_GPUf(source, target, s_currents, k, epsilon, t_direction, nThreads, mode)

        return out

    def _compToFields(self, comp, field):
        null = np.zeros(field.shape, dtype=complex)

        if comp == "Ex":
            field_c = fields(field, null, null, null, null, null)
        elif comp == "Ey":
            field_c = fields(null, field, null, null, null, null)
        elif comp == "Ez":
            field_c = fields(null, null, field, null, null, null)
        elif comp == "Hx":
            field_c = fields(null, null, null, field, null, null)
        elif comp == "Hy":
            field_c = fields(null, null, null, null, field, null)
        elif comp == "Hz":
            field_c = fields(null, null, null, null, null, field)

        return field_c

if __name__ == "__main__":
    print("System interface for POPPy.")
