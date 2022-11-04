from tiltGauss import tiltGauss
import matplotlib.pyplot as pt
import numpy as np

def getEffs(freq, w0):
    cl = 299792458e3
    lam = cl / (freq*1e9)
    eta_s_sec, eta_t, eta_ap = tiltGauss(lam, w0)
    return eta_s_sec, eta_t, eta_ap

if __name__ == "__main__":

    freqs = [218, 228, 238, 247, 257, 267, 277, 287, 297, 307, 317, 327]
    w0m = np.loadtxt("./personal/w0m.txt")

    eta_s_sec = []
    #eta_s_pri = []
    eta_t = []
    eta_ap = []

    for freq, w0 in zip(freqs, w0m):
        ess, et, eap = getEffs(freq, w0)

        eta_s_sec.append(ess)
        #eta_s_pri.append(esp)
        eta_t.append(et)
        eta_ap.append(eap)

    np.savetxt("eta_s_sec_gauss.txt", eta_s_sec)
    np.savetxt("eta_t_gauss.txt", eta_t)
    np.savetxt("eta_ap_gauss.txt", eta_ap)

    fig, ax = pt.subplots(1,1)
    ax.plot(freqs, eta_s_sec, label=r"$\eta^\mathrm{sec}_\mathrm{s}$")
    #ax.plot(freqs, eta_s_pri)
    ax.plot(freqs, eta_t, label=r"$\eta_\mathrm{t}$")
    ax.plot(freqs, eta_ap, label=r"$\eta_\mathrm{ap}$")
    ax.set_xlabel(r"$f$ / GHz")
    ax.set_ylabel(r"$\eta$")
    ax.legend(frameon=False, prop={'size': 10},handlelength=1)
    pt.show()
