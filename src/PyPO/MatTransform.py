import numpy as np
import PyPO.WorldParam as world

##
# @file
# Transformation formalism for PyPO.
#
# This script contains the methods for applying matrix transformations to objects..

##
# Generate a 3D rotation matrix and append to previous matrix.
# The appending is done by right matrix multiplication of the old transformation matrix with the rotation matrix.
#
# @param theta Numpy array of length 3 containing the xyz rotation angles.
# @param matAppend The previous matrix. Defaults to identity matrix.
# @param pivot Pivot for the rotation. Defaults to origin.
# @param radians Whether theta is in radians or degrees. Defaults to False (degrees).
#
# @returns matOut Full 4D affine transformation matrix.
def MatRotate(theta, matAppend=None, pivot=None, radians=False):
    pivot = world.ORIGIN() if pivot is None else pivot
    matAppend = world.INITM() if matAppend is None else matAppend

    if radians:
        theta_x, theta_y, theta_z = theta

    else:
        theta_x, theta_y, theta_z = np.radians(theta)

    ox, oy, oz = pivot

    trans1 = np.array([[1, 0, 0, -ox],
                    [0, 1, 0, -oy],
                    [0, 0, 1, -oz],
                    [0, 0, 0, 1]])

    rotX = np.array([[1, 0, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x), 0],
                    [0, np.sin(theta_x), np.cos(theta_x), 0],
                    [0, 0, 0, 1]])

    rotY = np.array([[np.cos(theta_y), 0, np.sin(theta_y), 0],
                    [0, 1, 0, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
                    [0, 0, 0, 1]])

    rotZ = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
                    [np.sin(theta_z), np.cos(theta_z), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    trans2 = np.array([[1, 0, 0, ox],
                    [0, 1, 0, oy],
                    [0, 0, 1, oz],
                    [0, 0, 0, 1]])
    
    matOut = (trans2 @ (rotZ @ (rotY @ (rotX @ trans1)))) @ matAppend

    return matOut

##
# Generate a 3D translation matrix and append to previous matrix.
# The appending is done by right matrix multiplication of the old transformation matrix with the translation matrix.
#
# @param trans Numpy array of length 3 containing the xyz translations, in mm.
# @param matAppend The previous matrix. Defaults to identity matrix.
#
# @returns matOut Full 4D affine transformation matrix.
def MatTranslate(trans, matAppend=None):
    matAppend = world.INITM() if matAppend is None else matAppend
    xt, yt, zt = trans
    trans = np.array([[1, 0, 0, xt],
                    [0, 1, 0, yt],
                    [0, 0, 1, zt],
                    [0, 0, 0, 1]])

    matOut = trans @ matAppend

    return matOut

##
# Invert a transformation matrix, both the rotational and translational part. 
#
# @param mat Full 4D affine transformation matrix.
#
# @returns matInv Full 4D affine inverse transformation matrix.
def InvertMat(mat):
    R_T = mat[:3, :3].T
    R_Tt = -R_T @ mat[:3, -1]

    matInv = world.INITM()
    matInv[:3, :3] = R_T
    matInv[:3, -1] = R_Tt

    return matInv

