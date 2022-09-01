import numpy as np 
import matplotlib.pyplot as pt

def plotPointing(ty=''):
    az_no_chop = np.load('personal/pointing/no_chop/az{}.npy'.format(ty))
    el_no_chop = np.load('personal/pointing/no_chop/el{}.npy'.format(ty))
    
    az_pos_1 = np.load('personal/pointing/pos_1/az{}.npy'.format(ty))
    el_pos_1 = np.load('personal/pointing/pos_1/el{}.npy'.format(ty))
    
    az_pos_2 = np.load('personal/pointing/pos_2/az{}.npy'.format(ty))
    el_pos_2 = np.load('personal/pointing/pos_2/el{}.npy'.format(ty))
    
    freqs = np.load('personal/pointing/freq_range.npy')
    
    az_rel_1 = az_pos_1 - az_no_chop
    az_rel_2 = az_pos_2 - az_no_chop
    
    mean_p1 = np.mean(az_rel_1)
    mean_p2 = np.mean(az_rel_2)
    
    std_p1 = np.std(az_rel_1)
    std_p2 = np.std(az_rel_2)
    
    print("Mean pointing offsets relative to no chopper: p1 = {} +- {} [as], p2 = {} +- {} [as]".format(mean_p1, std_p1, mean_p2, std_p2))
    
    fig, ax = pt.subplots(1,1)
    ax.plot(freqs, az_rel_1, color='blue', label='pos_1')
    ax.plot(freqs, az_rel_2, color='red', label='pos_2')
    ax.legend(frameon=False, prop={'size': 10}, handlelength=1)
    
    ax.set_title('pointing relative to no chopper')
    ax.set_ylabel(r'dAz / [as]')
    ax.set_xlabel(r'$f$ / [GHz]')
    
    pt.show()
    
if __name__ == "__main__":
    ty = '_APWS'
    plotPointing(ty)

