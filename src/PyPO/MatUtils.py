import numpy as np

##
# @file
# File containing a method for finding a connected subset, centered around a starting index, in a matrix.
# Used for creating masks for fitting Gaussians.

##
# Find extent of centered connected subset in a matrix.
#
# @param mat Matrix of which to find largest connected subset.
# @param component Value on which to differentiate subset.
# @param idx_start Matrix index on which to center the subset.
# 
# @returns Indices of row and column of limits of subset.
def findConnectedSubsets(mat, component, idx_start):
    row_start_upp = mat[idx_start[0],(idx_start[1]+1):]
    row_start_low = mat[idx_start[0],:idx_start[1]]
    
    col_start_upp = mat[(idx_start[0]+1):,idx_start[1]]
    col_start_low = mat[:idx_start[0],idx_start[1]]

    num_row_upp_conn = 0
    num_row_low_conn = 0
    num_col_upp_conn = 0
    num_col_low_conn = 0
    
    for i in row_start_upp:
        if i == component:
            num_row_upp_conn += 1
        else:
            break
    
    for i in np.flip(row_start_low):
        if i == component:
            num_row_low_conn += 1
        else:
            break

    for i in col_start_upp:
        if i == component:
            num_col_upp_conn += 1
        else:
            break
    
    for i in np.flip(col_start_low):
        if i == component:
            num_col_low_conn += 1
        else:
            break
   
    lims_row = np.array(range(idx_start[0] - num_row_low_conn, idx_start[0] + num_row_upp_conn + 1))
    lims_col = np.array(range(idx_start[1] - num_col_low_conn, idx_start[1] + num_col_upp_conn + 1))

    return lims_row, lims_col
