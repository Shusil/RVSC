# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:52:00 2016

@author: sxd7257
"""

import sys
sys.path.append("C://Users//sxd7257//Dropbox//Python Scripts")
import numpy as np
import scipy.io as sio
import myFunctions as func

for ind in range(0,16):
    ind = 12
    exec("fileName = 'rawData"+str(ind)+".mat'")
    matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\'+fileName)
#    matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TestSetMat\\'+fileName)
    keys = matContents.keys()
    for i in keys:
        exec(i+"= matContents['"+i+"']")
#    v = np.copy(func.stretchContrast(vol[:,:,:,0],0,99))
    v = np.copy(vol[:,:,:,0])
    gtS = np.copy(gt[:,:,:,0])
    func.displayMontage(v,5)
#    func.displayMontageRGB(v,gtS,5)
    