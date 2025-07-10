"""!
@file
File containing the optimiser used in PyPO for optimising 
"""

import numpy as np
from scipy import optimize

def runOptimiser():
    """!
    Optimise the configuration of a (group of) element(s), given: 
        a cost function 
        an initial ray frame 
        a planar terminal surface on which the final frame is calculated 
        and a list of degrees of freedom.
    """


