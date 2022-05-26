import src.Python.Beams as beams 

def testbeam():
    x_lims = [-100, 100]
    y_lims = [-100, 100]
    gridsize = [2001, 2001]
    pw = beams.PlaneWave(x_lims, y_lims, gridsize)
    pw.plotBeam()
    
if __name__ == "__main__":
    testbeam()
