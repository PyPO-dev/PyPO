import copy
import numpy as np

def copyGrid(grid):
    """
    Obtain a deep copy (i.e. a new object with a new address) of object.
    
    @param  ->
        grid        :   The object to be deepcopied
    """
    
    copy_grid = copy.deepcopy(grid)
    return copy_grid
    
    
