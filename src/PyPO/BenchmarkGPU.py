import numpy as np
import sys
import os

import matplotlib.pyplot as pt

from scipy.interpolate import bisplrep

def timePropagation(ns, nt, s):
    
    
    cl = 299792458e3
    lam = cl / (217 * 1e9)

    cpp_path = 'src/C++/'

    k = 2 * np.pi / lam

    # Initialize system

    center_cam = np.array([0,0,100])
    gridsize_cam = [nt[0], nt[1]]

    # Define uncorrected beam grid
    lims_x = [-70, 70]
    lims_y = [-70, 70]
    gridsize_beam = [ns[0], ns[1]]
    
    s.addBeam(lims_x, lims_y, gridsize_beam, beam='pw', pol=np.array([1,0,0]), amp=1, phase=0, comp='Ex', units='mm', cRot=np.zeros(3))
    
    # Add camera in secondary aperture
    s.addCamera(lims_x, lims_y, gridsize_cam, center=center_cam, name = "cam1", gmode='xy')

    s.inputBeam.calcJM(mode='PMC')

    s.addPlotter(save='../images/')

    s.initPhysOptics(target=s.system["cam1"], k=k, numThreads=11, cpp_path=cpp_path)
    time = s.runPhysOptics(save=1, material_source='alu', prec='single', device='gpu', ret=True)

    return time

def Benchmark_GPU(s):
    # Measure to 1501, extrapolate to 9501 later
    nt = [101, 501, 1001, 1501]
    ns = [101, 501, 1001, 1501]

    time = np.zeros((len(ns), len(nt)))
    '''
    for i, nns in enumerate(ns):
        for j, nnt in enumerate(nt):
        
            _nt = [nnt, nnt]
            _ns = [nns, nns]

            time[i,j] = timePropagation(_ns, _nt, s)
    '''
    time = np.loadtxt("benchmark.txt")
    #np.savetxt("benchmark.txt", time)
    
    # Calculate linear fits along x and y directions.
    # Extrapolate to some really big value, 9501
    
    # First, obtain time_nt = b(ns) + a(ns) * nt 
    # Then solve for: T(x,y) = bby + bay*y + (aby + aay*y) * x
    a_l = []
    b_l = []
    
    for i in range(4):
        a, b = np.polyfit(np.array([x**2 for x in nt]), time[i,:],1)
        a_l.append(a)
        b_l.append(b)
        
    x_test = np.linspace(0, 2501**2)
    y_test = np.linspace(0, 2501**2)
    
    func1 = np.poly1d([a_l[1], b_l[1]]) 
    func2 = np.poly1d([a_l[2], b_l[2]]) 
    func3 = np.poly1d([a_l[3], b_l[3]]) 
    '''
    pt.scatter([x**2 for x in nt], time[2,:])
    pt.plot(x_test, func2(x_test))
    pt.scatter([x**2 for x in nt], time[3,:])
    pt.plot(x_test, func3(x_test))
    pt.show()
    '''
    aay, aby = np.polyfit(np.array([x**2 for x in ns]), np.array(a_l),1)
    bay, bby = np.polyfit(np.array([x**2 for x in ns]), np.array(b_l),1)
    '''
    funca = np.poly1d([aay, aby]) 
    funcb = np.poly1d([bay, bby]) 
    
    ims = bby + bay*ns[1]**2 + (aby + aay*ns[1]**2) * nt[3]**2
    
    pt.scatter([x**2 for x in ns], a_l)
    pt.scatter([x**2 for x in ns], b_l)
    
    pt.plot(x_test, funca(x_test))
    pt.plot(x_test, funcb(x_test))
    pt.show()
    
    pt.scatter(nt[3]**2, ims)
    pt.plot(x_test, func1(x_test))
    pt.show()
    '''
    save_coeff = os.path.isdir('./save/benchmark/')
        
    if not save_coeff:
        os.makedirs('./save/benchmark/')
        
    os.system('nvidia-smi --query-gpu=gpu_name --format=csv > save/benchmark/gpu_name.temp')
    
    gpu_t = []
    with open('./save/benchmark/gpu_name.temp', 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                gpu_t = line
            
    gpu_t = gpu_t.replace(" ", "_") + "_GPU"
    gpu_t = gpu_t.replace("\n", "")
    
    coeff = [bby, bay, aby, aay]
    
    gpu_dict = {"name"      : gpu_t,
                "coeff"     : coeff}
            
    os.system('rm -rf save/benchmark/gpu_name.temp')
    
    '''
    coef0x = np.polyfit(np.array([x**2 for x in nt]), time[:-1,0],1)
    coef1x = np.polyfit(np.array([x**2 for x in nt]), time[:-1,1],1)
    coef2x = np.polyfit(np.array([x**2 for x in nt]), time[:-1,2],1)
    coef3x = np.polyfit(np.array([x**2 for x in nt]), time[:-1,3],1)
    
    # Determine slope in t_ns()

    poly1d_fn0x = np.poly1d(coef0x) 
    poly1d_fn1x = np.poly1d(coef1x) 
    poly1d_fn2x = np.poly1d(coef2x) 
    
    coef0y = np.polyfit(np.array([x**2 for x in ns]), time[0,:-1],1)
    coef1y = np.polyfit(np.array([x**2 for x in ns]), time[1,:-1],1)
    coef2y = np.polyfit(np.array([x**2 for x in ns]), time[2,:-1],1)
    coef3y = np.polyfit(np.array([x**2 for x in ns]), time[3,:-1],1)
    
    
    poly1d_fn0y = np.poly1d(coef0y) 
    poly1d_fn1y = np.poly1d(coef1y) 
    poly1d_fn2y = np.poly1d(coef2y) 
     
    Nmax = 9501
    
    nt = [101, 501, 1001, 1501, Nmax]
    ns = [101, 501, 1001, 1501]
     
    lim0x = poly1d_fn0x(Nmax**2)
    lim1x = poly1d_fn1x(Nmax**2)
    lim2x = poly1d_fn2x(Nmax**2)
    
    
    lim0y = poly1d_fn0y(Nmax**2)
    lim1y = poly1d_fn1y(Nmax**2)
    lim2y = poly1d_fn2y(Nmax**2)
    
    # Now replace last values with the extrapolated ones
    time[0,-1] = lim0y
    time[1,-1] = lim1y
    time[2,-1] = lim2y
    
    time[-1,0] = lim0x
    time[-1,1] = lim1x
    time[-1,2] = lim2x
    
    
    poly1d_fn3x = np.poly1d(coef3x) 
    
    
    poly1d_fn3y = np.poly1d(coef3y)
    
    lim3x = poly1d_fn3x(Nmax**2)
    lim3y = poly1d_fn3y(Nmax**2)

    time[-1,3] = (lim3y + lim3y) / 2
    
    grid_y, grid_x = np.meshgrid(np.array([x**2 for x in ns]), np.array([x**2 for x in nt]))
    tcks = bisplrep(grid_x.ravel(), grid_y.ravel(), time.ravel(), kx=3, ky=3, s=10)
    
    save_tcks = os.path.isdir('./save/benchmark/')
        
    if not save_tcks:
        os.makedirs('./save/benchmark/')
        
    os.system('nvidia-smi --query-gpu=gpu_name --format=csv > save/benchmark/gpu_name.temp')
    
    gpu_t = []
    with open('./save/benchmark/gpu_name.temp', 'r') as f:
        for i, line in enumerate(f):
            if i == 1:
                gpu_t = line
            
    gpu_t = gpu_t.replace(" ", "_") + "_GPU"
    gpu_t = gpu_t.replace("\n", "")
    
    tcks_l = []
    for tt in tcks:
        if isinstance(tt, np.ndarray):
            tt = tt.tolist()
            
        tcks_l.append(tt)
    
    gpu_dict = {"name"      : gpu_t,
                "tcks"      : tcks_l}
            
    os.system('rm -rf save/benchmark/gpu_name.temp')
    '''
    return gpu_dict
            
if __name__ == "__main__":
    print("Run the GPU benchmark from the System.py module.")
    
        
    
    
        

 
 

 
