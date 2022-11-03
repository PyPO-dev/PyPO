from tiltGauss import tiltGauss
import matplotlib.pyplot as pt
import numpy as np

def getEffs(freq, w0):
    cl = 299792458e3
    lam = cl / (freq*1e9)
    eta_s_sec, eta_s_pri, eta_t, eta_ap = tiltGauss(lam, w0)
    return eta_s_sec, eta_s_pri, eta_t, eta_ap

if __name__ == "__main__":

    freqs = [218, 228, 238, 247, 257, 267, 277, 287, 297, 307, 317, 327]
    w0m = np.loadtxt("./personal/w0m.txt")

    eta_s_sec = []
    eta_s_pri = []
    eta_t = []
    eta_ap = []

    for freq, w0 in zip(freqs, w0m):
        ess, esp, et, eap = getEffs(freq, w0)

        eta_s_sec.append(ess)
        eta_s_pri.append(esp)
        eta_t.append(et)
        eta_ap.append(eap)

    pt.plot(freqs, eta_s_sec)
    pt.plot(freqs, eta_s_pri)
    pt.plot(freqs, eta_t)
    pt.plot(freqs, eta_ap)
    pt.show()
