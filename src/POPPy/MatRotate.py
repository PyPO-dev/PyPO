import numpy as np

def MatRotate(theta, points, origin=np.zeros(3), vecRot=False, matOut=False, radians=False):
    
    """
    Create 3D rotation matrix and rotate grids of points.
    
    @param theta Array containing rotations around x,y,z axes in degrees.
    @param points Containers of 3D points to be rotated.
    @param origin Origin of rotation.
    @param vecRot Whether points or vector components are to be rotated.
    @param matOut Return full rotation matrix yes or no.
    @param radians Whether theta is in degrees or radians.
    
    @return pointsRot 3D rotated points.
    @return matOut Full rotation matrix, for taking inverse rotations: R^-1 = R^T.
    """
    if radians:
        theta_x, theta_y, theta_z = theta
    
    else:
        theta_x, theta_y, theta_z = np.radians(theta)
    
    px, py, pz = points
    
    rotX = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    
    rotY = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    rotZ = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    
    if vecRot:
        ox, oy, oz = np.zeros(3)
        rotAll = np.linalg.inv(np.matmul(rotZ, np.matmul(rotY, rotX))).T
        
    else:
        ox, oy, oz = origin
        rotAll = np.matmul(rotZ, np.matmul(rotY, rotX))

    tX = px - ox
    tY = py - oy
    tZ = pz - oz
        
    qx = ox + rotAll[0,0]*tX + rotAll[0,1]*tY + rotAll[0,2]*tZ
    qy = oy + rotAll[1,0]*tX + rotAll[1,1]*tY + rotAll[1,2]*tZ
    qz = oz + rotAll[2,0]*tX + rotAll[2,1]*tY + rotAll[2,2]*tZ
    
    pointsRot = np.array([qx, qy, qz])
    
    if matOut:
        return pointsRot, rotAll
    
    else:
        return pointsRot
    
