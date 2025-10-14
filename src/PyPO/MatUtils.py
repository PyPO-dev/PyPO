"""!
@file
File containing a method for finding a connected subset, centered around a starting index, in a matrix.
Used for creating masks for fitting Gaussians.
"""

import numpy as np

def findConnectedSubsets(mat, component, idx_start):
    """!
    Find extent of centered connected subset in a matrix.
    Note that this method is agnostic of the underlying coordinatisation, i.e the extent is given in indices w.r.t the supplied matrix itself.

    @param mat Matrix of which to find largest connected subset.
    @param component Value on which to differentiate subset.
    @param idx_start Matrix index on which to center the subset.

    @returns Indices of row and column of limits of subset.
    """

    row_start_upp = mat[idx_start[0],(idx_start[1]+1):]
    row_start_low = mat[idx_start[0],:idx_start[1]]
    
    col_start_upp = mat[(idx_start[0]+1):,idx_start[1]]
    col_start_low = mat[:idx_start[0],idx_start[1]]

    num_row_upp_conn = 0
    num_row_low_conn = 0
    num_col_upp_conn = 0
    num_col_low_conn = 0
    
    for i in row_start_upp:
        if i == component:
            num_row_upp_conn += 1
        else:
            break
    
    for i in np.flip(row_start_low):
        if i == component:
            num_row_low_conn += 1
        else:
            break

    for i in col_start_upp:
        if i == component:
            num_col_upp_conn += 1
        else:
            break
    
    for i in np.flip(col_start_low):
        if i == component:
            num_col_low_conn += 1
        else:
            break
   
    lims_row = np.array(range(idx_start[0] - num_row_low_conn, idx_start[0] + num_row_upp_conn + 1))
    lims_col = np.array(range(idx_start[1] - num_col_low_conn, idx_start[1] + num_col_upp_conn + 1))
    
    return lims_row, lims_col
    
def findRotation(self, v, u):
    """!
    Find rotation matrix to rotate v onto u.
    
    @param v Numpy array of length 3. 
    @param u Numpy array of length 3.
    """

    I = np.eye(3)
    if np.array_equal(v, u):
        return self.copyObj(world.INITM())

    lenv = np.linalg.norm(v)
    lenu = np.linalg.norm(u)

    if lenv == 0 or lenu == 0:
        self.clog.error("Encountered 0-length vector. Cannot proceed.")
        return None

    w = np.cross(v/lenv, u/lenu)

    lenw = np.linalg.norm(w)
    
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    R = I + K + K @ K * (1 - np.dot(v, u)) / lenw**2

    R_transf = self.copyObj(world.INITM())
    R_transf[:-1, :-1] = R
    
    return R_transf

def getAnglesFromMatrix(self, M):
    """!
    Find x, y and z rotation angles from general rotation matrix.
    Note that the angles are not necessarily the same as the original angles of the matrix.
    However, the matrix constructed by the found angles applies the same 3D rotation as the input matrix.
    
    @param M Numpy array of shape (3,3) containg a general rotation matrix.
    
    @returns r Numpy array of length 3 containing rotation angles around x, y and z.
    """

    if M[2,0] < 1:
        if M[2,0] > -1:
            ry = np.arcsin(-M[2,0])
            rz = np.arctan2(M[1,0], M[0,0])
            rx = np.arctan2(M[2,1], M[2,2])

        else:
            ry = np.pi / 2
            rz = -np.arctan2(-M[1,2], M[1,1])
            rx = 0

    else:
        ry = -np.pi / 2
        rz = np.arctan2(-M[1,2], M[1,1])
        rx = 0

    r = np.degrees(np.array([rx, ry, rz]))

    return r
