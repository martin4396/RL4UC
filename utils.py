import numpy as np
import pandas as pd
import torch

def matlab2pp(matlab_ans):
    gen_lst=np.array([0,4,8,14,15,21,22,23,24,30])
    sgen_lst=np.array([1,2,16,17,18,19,20,25,26,27,28,29,3,31,32,5,6,7,9,10,11,12])
    ext_grid_lst=np.array([13])
    matpower_idx=np.append(ext_grid_lst,np.append(gen_lst,sgen_lst))
    pp_idx=np.array([3,  0, 27, 31,  6,  7, 14, 15, 16, 17, 24,  1, 12, 23, 28, 29, 30,
        32,  2,  4,  5,  8,  9, 10, 11, 13, 18, 19, 20, 21, 22, 25, 26])
    idx_lst=pd.Series(matpower_idx,pp_idx).sort_index()
    return matlab_ans.loc[idx_lst]



