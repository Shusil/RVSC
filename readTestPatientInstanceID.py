# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:06:39 2016

@author: sxd7257
"""

import sys
import csv
sys.path.append("C://Users//sxd7257//Dropbox//Python Scripts")
import dicom
import os
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import groupby
import myFunctions as func
from skimage.draw import line
from vtk import *
import scipy.io as sio


# Root Dicom Directory
#dirPath = "N://ShusilDangi//RVSC//Test1Set"
dirPath = "N://ShusilDangi//RVSC//Test2Set"
dirList = []
for name in os.listdir(dirPath):
    if(os.path.isdir(os.path.join(dirPath,name))):
        dirList.append(os.path.join(dirPath,name))
dirList = natsorted(dirList)    # natural sorted list of dicom series

indices = range(0,16)
IOP = {}

for ind in indices[4:5]:
    print('Processing Patient'+str(ind))
    dicomPath = dirList[ind]
    lstFilesDCM = []
    lstFilesTXT = []
    files = []
    
    # list all the files in selected subdirectory
    for dirName, subdirList, fileList in os.walk(dicomPath):
        for filename in fileList:
            if ".dcm" in filename.lower():
                lstFilesDCM.append(os.path.join(dirName,filename))
            if "list.txt" in filename.lower():
                lstFilesTXT.append(os.path.join(dirName,filename))
    
    # read dicom info from all the files in selected subdirectory
    for lstFiles in lstFilesDCM:
        refDs = dicom.read_file(lstFiles)
        refDs.PatientName = lstFiles
        files.append(refDs)

    # Read all the files in the directory
    files.sort(key=lambda x:x.InstanceNumber,reverse=False)
    group = []
    gtGroup = []
    vol = []
    
    f = open(lstFilesTXT[0])
    contourFiles = f.read()
    f.close()

    for i in range(0,len(files),20):
        subgroup = []
        subVol = []
        gtSubGroup = []
        subGT = []
        for j in range(i,i+20):
            subVol.append(files[j])
            subgroup.append(files[j].pixel_array)
            img = np.zeros(files[j].pixel_array.shape)
            filePath = files[j].PatientName
            pattern = filePath[-1][:-4]
            if pattern in contourFiles:
                img = 255*np.ones(files[j].pixel_array.shape)
            gtSubGroup.append(img)                
        vol.append(subVol)
        group.append(subgroup)
        gtGroup.append(gtSubGroup)

    dist = []
    xySpacings = [] 
    for sg in vol:
        normal = np.cross(np.array(sg[0].ImageOrientationPatient[0:3]),np.array(sg[0].ImageOrientationPatient[3:6]))
        distance = np.dot(normal,np.array(sg[0].ImagePositionPatient))
        dist.append(distance)
        for phases in sg:
            xySpacings.append(list(np.asarray(phases.PixelSpacing)))
    dist = np.asarray(dist)
    sliceSpacings = dist[:-1]-dist[1:]
    roundSliceSpacings = np.round(sliceSpacings,0)
    [uniqueSliceSpacings,unique_inverse,counts] = np.unique(roundSliceSpacings,return_inverse=True,return_counts=True)
    indx = np.argsort(counts)
    if(len(uniqueSliceSpacings)>1):
        print("NON-UNIFORM SLICE SPACING!!!")
    actualSliceSpacing = np.round(np.mean(sliceSpacings[np.nonzero(unique_inverse==indx[-1])]),2)

    xySpacing = np.asarray(xySpacings)
    xSpacing = np.unique(xySpacing[:,0])
    ySpacing = np.unique(xySpacing[:,1])
    print(xSpacing,ySpacing)
    if(len(xSpacing)==1 and len(ySpacing)==1):
        spacing = tuple((xSpacing[0],ySpacing[0],actualSliceSpacing))
    else:
        print('Slice spacing different in XY for different slices')    
        spacing = tuple((xSpacing[0],ySpacing[0],actualSliceSpacing))
    #spacing = tuple(xySpacing)+(actualSliceSpacingsList[maxInd],)
    print(spacing)

    vol4D = np.asarray(group)
    volXYZT = np.transpose(vol4D,(3,2,0,1))
    gt4D = np.asarray(gtGroup)
    gtXYZT = np.transpose(gt4D,(3,2,0,1))
    
    fig = plt.figure()
    ims = []
    for i in range(volXYZT.shape[3]):
#        montageImg = func.createMontage(volXYZT[:,:,:,i],4)
        montageImg = func.createMontageRGB(volXYZT[:,:,:,i],gtXYZT[:,:,:,i],4)
        im = plt.imshow(montageImg,cmap=plt.cm.gray,animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=100,blit=False,repeat_delay=100)
    plt.show()

    rawPatientData = {}
    rawPatientData.update({'vol': volXYZT})
    rawPatientData.update({'gt': gtXYZT})
    rawPatientData.update({'spacing':spacing})
    rawPatientData.update({'IOP':np.asarray(vol[0][0].ImageOrientationPatient)[0:6]})
#    exec("sio.savemat('N://ShusilDangi//RVSC//Test1SetMat//rawData"+str(ind+32)+".mat',rawPatientData)")

# Test Set 1:
# 10 -> IPP misaligned in two slices
# 14 -> IPP slightly misaligned, different xy-spacings

# Test Set 2:
# 4 -> Slice spacing mismatch, 1 slice missing
# 10 -> Slice spacing mismatch, 1 slice missing
