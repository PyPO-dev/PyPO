## 
# @file
# Definitions of internal PyPO data structures.

##
# Base class for EH fields and JM currents.
class resContainer(object):

    ##
    # Constructor. Takes EH/JM components and assigns them to member variables.
    # Also creates a member variable, in which the EH/JM labels are stored for the getter/setter functions. 
    #
    # @param args Sequence of EH/JM components.
    # @param restype Whether object is a field ("EH") or a current ("JM"). Default is "EH"
    def __init__(self, *args, restype=None):
        self.type = "EH" if restype is None else restype
        self.memlist = []

        n = 0
        ax = ["x", "y", "z"]
        
        for i, arg in enumerate(args):
            if not i % 3 and i != 0:
                n += 1
            self.memlist.append(f"{self.type[n]}{ax[i - 3*n]}")
            setattr(self, self.memlist[i], arg)
    
    ##
    # Set EH/JM metadata.
    #
    # @param surf Name of surface on which EH/JM are defined.
    # @param k Wavenumber in 1 / mm of EH/JM.
    def setMeta(self, surf, k):
        self.surf = surf 
        self.k = k

    ##
    # Get EH/JM component.
    #
    # @param idx Index of EH/JM component in memberlist.
    def __getitem__(self, idx):
        try:
            return getattr(self, self.memlist[idx])
        
        except:
            raise IndexError(f'Index out of range: {idx}')

    ##
    # Set EH/JM component.
    #
    # @param idx Index of EH/JM component in memberlist.
    # @param item Component to set in object.
    def __setitem__(self, idx, item):
        try:
            setattr(self, self.memlist[idx], item)
        
        except:
            raise IndexError(f'Index out of range: {idx}')

    def T(self):
        for i in range(6):
            self[i] = self[i].T
        return self
    
    def H(self):
        for i in range(6):
            self[i] = self[i].conj().T
        return self

##
# Wrapper for making a currents object. 
class currents(resContainer):
    ##
    # Constructor. Takes JM components and assigns them to member variables.
    #
    # @param Jx J-current x-component.
    # @param Jy J-current y-component.
    # @param Jz J-current z-component.
    # @param Mx M-current x-component.
    # @param My M-current y-component.
    # @param Mz M-current z-component.
    def __init__(self, Jx, Jy, Jz, Mx, My, Mz):
        super().__init__(Jx, Jy, Jz, Mx, My, Mz, restype="JM")

##
# Wrapper for making a fields object. 
class fields(resContainer):
    ##
    # Constructor. Takes JM components and assigns them to member variables.
    #
    # @param Ex E-field x-component.
    # @param Ey E-field y-component.
    # @param Ez E-field z-component.
    # @param Hx H-field x-component.
    # @param Hy H-field y-component.
    # @param Hz H-field z-component.
    def __init__(self, Ex, Ey, Ez, Hx, Hy, Hz):
        super().__init__(Ex, Ey, Ez, Hx, Hy, Hz, restype="EH")

class rfield(object):
    def __init__(self, Prx, Pry, Prz):
        self.x = Prx
        self.y = Pry
        self.z = Prz

##
# Structure for storing scalar fields and associated metadata.
class scalarfield(object):
    ##
    # Constructor for scalar field.
    #
    # @param S Scalar field.
    def __init__(self, S):
        self.S = S

    ##
    # Set scalar field metadata.
    #
    # @param surf Name of surface on which scalar field is defined.
    # @param k Wavenumber in 1 / mm of scalar field.
    def setMeta(self, surf, k):
        self.surf = surf 
        self.k = k

##
# Structure for storing reflector grids, area and normals
class reflGrids(object):
    def __init__(self, x, y, z, nx, ny, nz, area):
        self.x = x
        self.y = y
        self.z = z

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.area = area

class frame(object):
    def __init__(self, size, x, y, z, dx, dy, dz):
        self.size = size
        self.x = x
        self.y = y
        self.z = z

        self.dx = dx
        self.dy = dy
        self.dz = dz

