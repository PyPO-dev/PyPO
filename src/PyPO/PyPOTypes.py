"""!
@file
Definitions of PyPO data structures.
""" 

class resContainer(object):
    """!
    Base class for EH fields and JM currents.

    @ingroup public_api_types
    """
    
    def __init__(self, *args, restype=None):
        """!
        Constructor. Takes EH/JM components and assigns them to member variables.
        Also creates a member variable, in which the EH/JM labels are stored for the getter/setter functions. 
        
        @param args Sequence of EH/JM components.
        @param restype Whether object is a field ("EH") or a current ("JM"). Default is "EH".
        """

        self.type = "EH" if restype is None else restype
        self.memlist = []
        self.shape = args[0].shape
        self.size = self.shape[0] * self.shape[1]
        
        n = 0
        ax = ["x", "y", "z"]
        
        for i, arg in enumerate(args):
            if not i % 3 and i != 0:
                n += 1
            self.memlist.append(f"{self.type[n]}{ax[i - 3*n]}")
            setattr(self, self.memlist[i], arg)
    
    def setMeta(self, surf, k):
        """!
        Set EH/JM metadata.
        
        @param surf Name of surface on which EH/JM are defined.
        @param k Wavenumber in 1 / mm of EH/JM.
        """

        self.surf = surf 
        self.k = k

    def __getitem__(self, idx):
        """!
        Get EH/JM component.
        
        @param idx Index of EH/JM component in memberlist.
        """

        try:
            return getattr(self, self.memlist[idx])
        
        except:
            raise IndexError(f'Index out of range: {idx}')

    def __setitem__(self, idx, item):
        """!
        Set EH/JM component.
        
        @param idx Index of EH/JM component in memberlist.
        @param item Component to set in object.
        """

        try:
            setattr(self, self.memlist[idx], item)
        
        except:
            raise IndexError(f'Index out of range: {idx}')

    def T(self):
        """!
        Transpose of own fields/currents.
        """

        for i in range(6):
            self[i] = self[i].T
        return self
    
    def H(self):
        """!
        Complex conjugate (Hermitian) transpose of own fields/currents.
        """

        for i in range(6):
            self[i] = self[i].conj().T
        return self

class currents(resContainer):
    """!
    Wrapper for making a currents object. 

    @ingroup public_api_types
    """

    def __init__(self, Jx, Jy, Jz, Mx, My, Mz):
        """!
        Constructor. Takes JM components and assigns them to member variables.
        
        @param Jx J-current x-component.
        @param Jy J-current y-component.
        @param Jz J-current z-component.
        @param Mx M-current x-component.
        @param My M-current y-component.
        @param Mz M-current z-component.
        """

        super().__init__(Jx, Jy, Jz, Mx, My, Mz, restype="JM")

class fields(resContainer):
    """!
    Wrapper for making a fields object. 

    @ingroup public_api_types
    """

    def __init__(self, Ex, Ey, Ez, Hx, Hy, Hz):
        """!
        Constructor. Takes JM components and assigns them to member variables.
        
        @param Ex E-field x-component.
        @param Ey E-field y-component.
        @param Ez E-field z-component.
        @param Hx H-field x-component.
        @param Hy H-field y-component.
        @param Hz H-field z-component.
        """

        super().__init__(Ex, Ey, Ez, Hx, Hy, Hz, restype="EH")

class rfield(object):
    """!
    Class for making a real-vaLuad 3D object, used for Poynting vectors.

    @ingroup public_api_types
    """

    def __init__(self, Prx, Pry, Prz):
        """!
        Constructor. Takes Poynting components and assigns to member variables.
        
        @param Prx Poynting x-component.
        @param Pry Poynting y-component.
        @param Prz Poynting z-component.
        """

        self.x = Prx
        self.y = Pry
        self.z = Prz

class scalarfield(object):
    """!
    Structure for storing scalar fields and associated metadata.

    @ingroup public_api_types
    """

    def __init__(self, S):
        """!
        Constructor for scalar field.
        
        @param S Scalar field.
        """

        self.S = S

    def setMeta(self, surf, k):
        """!
        Set scalar field metadata.
        
        @param surf Name of surface on which scalar field is defined.
        @param k Wavenumber in 1 / mm of scalar field.
        """

        self.surf = surf 
        self.k = k

class reflGrids(object):
    """!
    Structure for storing reflector grids, area and normals

    @ingroup public_api_types
    """

    def __init__(self, x, y, z, nx, ny, nz, area):
        """!
        Constructor. Takes grid points, normals and area elements and stores them.
        
        @param x The x-components of the points making up the grid.
        @param y The y-components of the points making up the grid.
        @param z The z-components of the points making up the grid.
        @param nx The x-components of the normals to the grid.
        @param ny The y-components of the normals to the grid.
        @param nz The z-components of the normals to the grid.
        @param area The area elements of the grid.
        """

        self.x = x
        self.y = y
        self.z = z

        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.area = area

class frame(object):
    """!
    Structure for storing ray-trace frames.

    @ingroup public_api_types
    """

    def __init__(self, size, x, y, z, dx, dy, dz):
        """!
        Constructor. Takes frame points and directions and stores them.
        
        @param size Number of rays in frame.
        @param x The x-components of the rays in the frame.
        @param y The y-components of the rays in the frame.
        @param z The z-components of the rays in the frame.
        @param dx The x-components of the direction of the rays in the frame. 
        @param dy The y-components of the direction of the rays in the frame.
        @param dz The z-components of the direction of the rays in the frame.
        """

        self.size = size
        self.x = x
        self.y = y
        self.z = z

        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.snapshots = {}

    def setMeta(self, pos, ori, transf):
        """!
        Set frame metadata.
        
        @param pos Co-ordinate of reference point of frame..
        @param ori Reference orientation of frame.
        @param transf Transformation matrix for the frame.
        """

        self.pos = pos
        self.ori = ori
        self.transf = transf

