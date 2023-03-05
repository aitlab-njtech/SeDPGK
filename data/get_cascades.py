import pdb
import torch
import numpy as np
import os
# from pathlib import Path


def load_cascades(cascade_dir, device, dataset, trans=False, final=False):
    cas = []
    if final:
        cas.append(np.genfromtxt(cascade_dir.parent.joinpath('output.txt')))
    else:
        cas.append(np.genfromtxt('C:/Users/Ye Yue/Desktop/softlable/Proposed/'+dataset+'_'+str(3)+'.txt'))
#'C:/Users/Ye Yue/Desktop/softlable/Proposed/'+dataset+'_'+str(1)+'.txt'
    cas = torch.FloatTensor(cas)
    # pdb.set_trace()
    if trans:
        cas = torch.transpose(cas, 1, 2)
    cas = cas.to(device)
    return cas


def remove_overfitting_cascades(cascade_dir, patience):
    cas_list = os.listdir(cascade_dir)
    cas_list.sort(key=lambda x: int(x[:-4]))
    for i in range(patience):
        os.remove(cascade_dir.joinpath(cas_list[-1-i]))
