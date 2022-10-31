class currents(object):
    def __init__(self, Jx, Jy, Jz, Mx, My, Mz):
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz

        self.Mx = Mx
        self.My = My
        self.Mz = Mz

    def __getitem__(self, i):
        if i == 0:
            return self.Jx
        elif i == 1:
            return self.Jy
        elif i == 2:
            return self.Jz

        elif i == 3:
            return self.Mx
        elif i == 4:
            return self.My
        elif i == 5:
            return self.Mz

        raise IndexError('Index out of range: {}'.format(i))

class fields(object):
    def __init__(self, Ex, Ey, Ez, Hx, Hy, Hz):
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez

        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz

    def __getitem__(self, i):
        if i == 0:
            return self.Ex
        elif i == 1:
            return self.Ey
        elif i == 2:
            return self.Ez

        elif i == 3:
            return self.Hx
        elif i == 4:
            return self.Hy
        elif i == 5:
            return self.Hz

        raise IndexError('Index out of range: {}'.format(i))

class rfield(object):
    def __init__(self, Prx, Pry, Prz):
        self.Prx = Prx
        self.Pry = Pry
        self.Prz = Prz

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
