import numpy as np
import matplotlib.pyplot as pt

from src.POPPy.System import System

D = 5e5

def transmitter(sysObject):
    transmitter = {
            "name"      : "transmitter",
            "gmode"     : "uv",
            "gcenter"   : np.array([0, 0]),
            "flip"      : False,
            "lims_u"    : np.array([0, 2e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([501, 501]),
            "pmode"     : "focus",
            "ecc_uv"    : 0.5,
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0, 0, 1e3])
            }

    sysObject.addParabola(transmitter)

def receiver(sysObject):
    receiver = {
            "name"      : "receiver",
            "gmode"     : "uv",
            "gcenter"   : np.array([0, 0]),
            "flip"      : False,
            "lims_u"    : np.array([0, 2e3]),
            "lims_v"    : np.array([0, 360]),
            "gridsize"  : np.array([501, 501]),
            "pmode"     : "focus",
            "vertex"    : np.zeros(3),
            "focus_1"   : np.array([0, 0, 3e3])
            }

    sysObject.addParabola(receiver)
    sysObject.rotateGrids("receiver", np.array([0, 180, 0]))
    sysObject.translateGrids("receiver", np.array([0, 0, D]))

def source_plane(sysObject):
    source = {
            "name"      : "source",
            "gmode"     : "xy",
            "flip"      : True,
            "lims_x"    : np.array([-0.1, 0.1]),
            "lims_y"    : np.array([-0.1, 0.1]),
            "gridsize"  : np.array([3, 3]),
            }
    
    sysObject.addPlane(source)
    sysObject.rotateGrids("source", np.array([0, -31.46, 0]))
    sysObject.translateGrids("source", np.array([0, 0, 1e3]))

def target_plane(sysObject):
    target = {
            "name"      : "target",
            "gmode"     : "xy",
            "flip"      : False,
            "lims_x"    : np.array([-50, 50]),
            "lims_y"    : np.array([-50, 50]),
            "gridsize"  : np.array([301, 301]),
            }
    
    sysObject.addPlane(target)
    sysObject.rotateGrids("target", np.array([0, 0, 0]))
    sysObject.translateGrids("target", np.array([0, 0, D - 3e3]))

def trans_ff(sysObject):
    trans_ff = {
            "name"      : "trans_ff",
            "gmode"     : "AoE",
            "flip"      : False,
            "lims_Az"   : np.array([-30, 30]),
            "lims_El"   : np.array([-30, 30]),
            "gridsize"  : np.array([301, 301])
            }

    sysObject.addPlane(trans_ff)

def Offcenter_transmit_receive():
    s = System()

    transmitter(s)
    receiver(s)
    source_plane(s)
    target_plane(s)
    trans_ff(s)

    grid_trans = s.generateGrids("transmitter")

    s.plotSystem(select=["receiver", "target"])

    PS = s.generatePointSource("source")
    JM = s.calcCurrents("source", PS)

    nThreads = 256
    device = "GPU"

    lam = 10

    source_to_transmitter = {
            "s_name"    : "source",
            "t_name"    : "transmitter",
            "s_current" : JM,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "JMEH"
            }

    JM1, EH1 = s.runPO(source_to_transmitter)
    s.plotBeam2D("transmitter", EH1.Ex)

    trans_to_ff = {
            "s_name"    : "transmitter",
            "t_name"    : "trans_ff",
            "s_current" : JM1,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "FF"
            }

    EH_trans_ff = s.runPO(trans_to_ff)
    s.plotBeam2D("trans_ff", EH_trans_ff.Ex)

    trans_to_receive = {
            "s_name"    : "transmitter",
            "t_name"    : "receiver",
            "s_current" : JM1,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "JMEH"
            }

    JM_receive, EH_receive = s.runPO(trans_to_receive)
    s.plotBeam2D("receiver", EH_receive.Ex)
    
    receive_to_target = {
            "s_name"    : "receiver",
            "t_name"    : "target",
            "s_current" : JM_receive,
            "lam"       : lam,
            "epsilon"   : 10,
            "exp"       : "fwd",
            "nThreads"  : nThreads,
            "device"    : device,
            "mode"      : "EH"
            }

    EH_target = s.runPO(receive_to_target)
    s.plotBeam2D("target", EH_target.Ex)

if __name__ == "__main__":
    Offcenter_transmit_receive()

