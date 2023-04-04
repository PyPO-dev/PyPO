import numpy as np
import matplotlib.pyplot as pt

def findConnectedSubsets(mat, component, idx_start):
    row_start_upp = mat[idx_start[0],(idx_start[1]+1):]
    row_start_low = mat[idx_start[0],:idx_start[1]]
    
    col_start_upp = mat[(idx_start[0]+1):,idx_start[1]]
    col_start_low = mat[:idx_start[0],idx_start[1]]

    #row_conn = np.argwhere(row_start.)

    #print(row_start_upp)
    #print(row_start_low)
    #print(col_start_upp)
    #print(col_start_low)

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

    #print(mat)
    #print(lims_row)
    #print(lims_col)
    #pt.imshow(mat)
    #pt.scatter(idx_start[1], idx_start[0], zorder=100, color="red")
    #pt.scatter(lims_col[0], idx_start[0], zorder=100, color="blue")
    #pt.scatter(lims_col[1], idx_start[0], zorder=100, color="blue")
    #pt.scatter(idx_start[1], lims_row[0], zorder=100, color="blue", marker="d")
    #pt.scatter(idx_start[1], lims_row[1], zorder=100, color="blue", marker="d")
    #pt.show()

    return lims_row, lims_col


if __name__ == "__main__":
    mat = np.zeros((6,6))
    mat[1,2] = 1
    mat[0,2] = 1
    mat[2,2] = 1
    mat[1,3] = 1
    mat[1,1] = 1

    mat[5,5] = 1
    mat[5,4] = 1
    mat[5,3] = 1
    mat[5,2] = 1
    
    idx_start = [1,2]
    component = 1

    findConnectedSubsets(mat, component, idx_start)
