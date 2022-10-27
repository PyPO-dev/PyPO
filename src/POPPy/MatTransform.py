import numpy as np

def MatRotate(theta, matAppend, origin=np.zeros(3), radians=False):
    """
    Create 3D rotation matrix and rotate grids of points.

    @param  ->
        theta       :   Array containing rotations around x,y,z axes in degrees.
        origin      :   Origin of rotation.
        radians     :   Whether theta is in degrees or radians.

    @return ->
        matOut      :   Full affine rotation matrix.
    """

    if radians:
        theta_x, theta_y, theta_z = theta

    else:
        theta_x, theta_y, theta_z = np.radians(theta)

    ox, oy, oz = origin

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

    matOut = np.matmul(trans2, np.matmul(rotZ, np.matmul(rotY, np.matmul(rotX, trans1))))

    return np.matmul(matOut, matAppend)

def MatTranslate(trans, matAppend):
    xt, yt, zt = trans
    trans = np.array([[1, 0, 0, -xt],
                    [0, 1, 0, -yt],
                    [0, 0, 1, -zt],
                    [0, 0, 0, 1]])

    return np.matmul(trans, matAppend)
