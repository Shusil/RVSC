# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:19:23 2016

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
import scipy.ndimage.morphology as morph


def truncate(arr,decimal):
    # takes an array as input and truncates each element to the given decimal places
    # error if input is not an array
    arr = arr*10**decimal
    arr = np.trunc(arr).astype(int)
    arr[arr==0]=0
    truncArr = arr.astype(float)/10**decimal
    return truncArr
    

# Root Dicom Directory
dirPath = "N://ShusilDangi//RVSC//TrainingSet"
dirList = []
for name in os.listdir(dirPath):
    if(os.path.isdir(os.path.join(dirPath,name))):
        dirList.append(os.path.join(dirPath,name))
dirList = natsorted(dirList)    # natural sorted list of dicom series

indices = range(0,16)
IOP = {}

for ind in indices:
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
            if "manual.txt" in filename.lower():
                lstFilesTXT.append(os.path.join(dirName,filename))
    
    # read dicom info from all the files in selected subdirectory
    for lstFiles in lstFilesDCM:
        refDs = dicom.read_file(lstFiles)
        refDs.PatientName = lstFiles
        files.append(refDs)
    
    groups = []
    uniquekeys = []
    decimalPlaces = 4
#    if ind==69:
#        decimalPlaces = 1
    # Group images according to image orientation (patient)
    files.sort(key=lambda x: str(truncate(np.asarray(x.ImageOrientationPatient),decimalPlaces)), reverse=False)  
    for keyOrient, groupOrient in groupby(files, lambda x:str(truncate(np.asarray(x.ImageOrientationPatient),decimalPlaces))):
        groupsOrientation = []
        uniquekeysOrientation = []
        group = list(groupOrient)
        group.sort(key=lambda x:x.ImagePositionPatient, reverse=False)
        # Group patients according to image position patient
        for keyIPP, groupIPP in groupby(group,lambda x:x.ImagePositionPatient):
            sortedGroup = list(groupIPP)
            sortedGroup.sort(key=lambda x:x.InstanceNumber, reverse=False)
            groupsOrientation.append(sortedGroup)
            uniquekeysOrientation.append(keyIPP)
        groups.append(groupsOrientation)
        uniquekeys.append(uniquekeysOrientation)
    
    # Check for the group lengths
    grpLenth = []
    for i in groups: grpLenth.append(len(i))
    print grpLenth
    
    
    # handling the cases with duplicate slices
    numOfSlices = []
    for group in groups:
        for slices in group:
            numOfSlices.append(len(slices))
    requiredNumOfSlices = np.min(numOfSlices)
    
    # Get rid of duplicate slices in dicom series    
    correctedGroups = []
    for group in groups:
        correctedGroup = []
        for slices in group:
            numOfSlices.append(len(slices))
        for slices in group:
            if(len(slices) != requiredNumOfSlices):
                # There are duplicate slices, keep the ones acquired latest
                seriesTime = []
                groupSeries = []
                slices.sort(key=lambda x:x.SeriesTime,reverse=True)
                for keySeriesTime, groupSeriesTime in groupby(slices,lambda x:x.SeriesTime):
                    groupSameSTime = list(groupSeriesTime)
                    seriesTime.append(keySeriesTime)
                    groupSeries.append(groupSameSTime)
                correctedGroup.append(groupSeries[0])
            else:
                correctedGroup.append(slices)
        correctedGroups.append(correctedGroup)
    
    
    # Sort images according to their position from apex to base
    sortedGroups = []
    actualSliceSpacingsList = []
    #listedSliceSpacingsList = []
    for group in correctedGroups:
        if(len(group)>1):
            dist = []
            sortedGroup=[]
            listedSliceSpacing=[]
            for subgroup in group:
                normal = np.cross(np.array(subgroup[0].ImageOrientationPatient[0:3]),np.array(subgroup[0].ImageOrientationPatient[3:6]))
                distance = np.dot(normal,np.array(subgroup[0].ImagePositionPatient))
                dist.append(distance)
    #        for i in group: listedSliceSpacing.append(float(i[0].SpacingBetweenSlices))
            dist = np.asarray(dist)
            indxs = list(reversed(list(np.argsort(dist))))
            indxCorrected = list(reversed(list(np.argsort(dist))))
            sortedDist = dist[indxs]
            sliceSpacings = sortedDist[:-1]-sortedDist[1:]
            roundSliceSpacings = np.round(sliceSpacings,0)
            [uniqueSliceSpacings,unique_inverse,counts] = np.unique(roundSliceSpacings,return_inverse=True,return_counts=True)
            indx = np.argsort(counts)
    #        # Get rid of duplicate slices (very small distance)
    #        for i in range(len(roundSliceSpacings)):
                
            # correct for non-uniform slice spacing due to duplicate slices             
            if(len(uniqueSliceSpacings)>1):
                print("Slice Spacing Non-Uniform")
                print(sliceSpacings)
#                [uniqueSliceSpacings,unique_inverse,counts] = np.unique(roundSliceSpacings,return_inverse=True,return_counts=True)
#                indx = np.argsort(counts)
                reqSpacing = uniqueSliceSpacings[indx[-1]]   # This is the correct slice spacing
                for i in range(len(roundSliceSpacings)-1):
                    if(roundSliceSpacings[i]!=reqSpacing):
                        if(roundSliceSpacings[i]==0):
                            if(float(group[indxs[i]][0].SeriesTime)<float(group[indxs[i+1]][0].SeriesTime)):
                                indxCorrected.remove(indxs[i])
                            else:
                                indxCorrected.remove(indxs[i+1])     # Retain the slice acquired latest
                        elif((sliceSpacings[i]+sliceSpacings[i+1]-reqSpacing)<1):
                            indxCorrected.remove(indxs[i+1])    # Remove extra slice

            # Handling special case of two slices
            if(len(indxs)==2 and np.sum(roundSliceSpacings)==0):
                print("Duplicate Slices")
                print(sliceSpacings)
                if(float(group[indxs[0]][0].SeriesTime)<float(group[indxs[1]][0].SeriesTime)):
                    indxCorrected.remove(indxs[0])
                else:
                    indxCorrected.remove(indxs[1])     # Retain the slice acquired latest

            newDist = dist[indxCorrected]
            newSliceSpacings = newDist[1:]-newDist[:-1]
    #        print(listedSliceSpacing)
            print(newSliceSpacings)
#            actualSliceSpacing = np.round(np.mean(sliceSpacings),2)
            actualSliceSpacing = np.round(np.mean(sliceSpacings[np.nonzero(unique_inverse==indx[-1])]),2)
            actualSliceSpacingsList.append(actualSliceSpacing)
    #        meanListedSliceSpacing = np.round(np.mean(listedSliceSpacing),2)
    #        listedSliceSpacingsList.append(meanListedSliceSpacing)
            for i in indxCorrected:
                sortedGroup.append(group[i])
            sortedGroups.append(sortedGroup)
        else:
            actualSliceSpacingsList.append(0)
    #        for i in group: listedSliceSpacing.append(float(i[0].SpacingBetweenSlices))
    #        meanListedSliceSpacing = np.round(np.mean(listedSliceSpacing),2)
    #        listedSliceSpacingsList.append(meanListedSliceSpacing)
            sortedGroups.append(group)
    

    # Check for the group lengths
    grpLenth = []
    for i in sortedGroups: grpLenth.append(len(i))
    print grpLenth


#    lstFilesTXT = natsorted(lstFilesTXT)
    gtGroups = []
    endoGroups = []
    for group in sortedGroups:
        groupSlices = []
        endoSlices = []
        for slices in group:
            groupPhases = []
            endoPhases = []
            for phases in slices:
                img = np.zeros(phases.pixel_array.shape)
                endo = np.zeros(phases.pixel_array.shape)
                bp = np.zeros(phases.pixel_array.shape)
                filePath = phases.PatientName
                pattern = filePath[-1][:-4]
                for fileName in lstFilesTXT:
                    if pattern in fileName:
                        data = []
                        csvfile = open(fileName,'rb')
                        reader = csv.reader(csvfile, delimiter=' ',quotechar='|')
                        for row in reader:
                            data.append([float(row[0]),float(row[1])])
                        csvfile.close()
                        data = np.round(np.asarray(data)).astype(int)
                        for i in range(data.shape[0]-1):
                            rr,cc = line(data[i][0],data[i][1],data[i+1][0],data[i+1][1])
#                            rr[rr<0] = 0
#                            rr[rr>=img.shape[0]] = img.shape[0]-1
#                            cc[cc<0] = 0
#                            cc[cc>=img.shape[1]] = img.shape[1]-1
                            img[cc,rr]=255
                    if pattern+'-i' in fileName:
                        data = []
                        csvfile = open(fileName,'rb')
                        reader = csv.reader(csvfile, delimiter=' ',quotechar='|')
                        for row in reader:
                            data.append([float(row[0]),float(row[1])])
                        csvfile.close()
                        data = np.round(np.asarray(data)).astype(int)
                        for i in range(data.shape[0]-1):
                            rr,cc = line(data[i][0],data[i][1],data[i+1][0],data[i+1][1])
#                            rr[rr<0] = 0
#                            rr[rr>=img.shape[0]] = img.shape[0]-1
#                            cc[cc<0] = 0
#                            cc[cc>=img.shape[1]] = img.shape[1]-1
                            endo[cc,rr]=255
                        bp = morph.binary_fill_holes(endo)
                groupPhases.append(img)
                endoPhases.append(bp)
            groupSlices.append(groupPhases)
            endoSlices.append(endoPhases)
        gtGroups.append(groupSlices)
        endoGroups.append(endoSlices)
    
    for group in sortedGroups:
        print(group[0][0].ImageOrientationPatient)
    
    groupLen = []
    for group in sortedGroups:
        groupLen.append(len(group))    
    maxInd = np.argmax(groupLen)
    #maxInd = 4
    xySpacings = [] 
    saList = sortedGroups[maxInd]
    gtList = gtGroups[maxInd]
    bpList = endoGroups[maxInd]
    saVol = []
    for slices in saList:
        saSlices = []
        for phases in slices:
            saSlices.append(phases.pixel_array)
            xySpacings.append(list(np.asarray(phases.PixelSpacing)))
        saVol.append(saSlices)
    gtVol = []
    for slices in gtList:
        saSlices = []
        for phases in slices:
            saSlices.append(phases)
        gtVol.append(saSlices)

    endoVol = []
    for slices in bpList:
        saSlices = []
        for phases in slices:
            saSlices.append(phases)
        endoVol.append(saSlices)

    xySpacing = np.asarray(xySpacings)
    xSpacing = np.unique(xySpacing[:,0])
    ySpacing = np.unique(xySpacing[:,1])
    print(xSpacing,ySpacing)
    if(len(xSpacing)==1 and len(ySpacing)==1):
        spacing = tuple((xSpacing[0],ySpacing[0],actualSliceSpacingsList[maxInd]))
    else:
        print('Slice spacing different in XY for diffrent slices')    
        spacing = tuple((xSpacing[0],ySpacing[0],actualSliceSpacingsList[maxInd]))
    #spacing = tuple(xySpacing)+(actualSliceSpacingsList[maxInd],)
    print(spacing)
    
    vol4D = np.asarray(saVol)
    volXYZT = np.transpose(vol4D,axes=(3,2,0,1))
#    print(vol4D.shape)
    gt4D = np.asarray(gtVol)#.astype(np.uint8)
    gtXYZT = np.transpose(gt4D,axes=(3,2,0,1))

    bp4D = np.asarray(endoVol)
    bpXYZT = np.transpose(bp4D,axes=(3,2,0,1))

    volPh1 = volXYZT[:,:,:,0]
#    func.displayMontage(volPh1,4)
    gtPh1 = gtXYZT[:,:,:,0]
    bpPh1 = bpXYZT[:,:,:,0]
##    volS1 = vol4D[:,:,0,:]
#    func.displayMontageRGB(volPh1,gtPh1,4)
#    func.displayMontageRGB(volPh1,255*bpPh1,4)
#    func.displayMontageRGB(gtPh1,255*bpPh1,4)

#    fig = plt.figure()
#    ims = []
#    for i in range(volXYZT.shape[3]):
##        montageImg = func.createMontageRGB(volXYZT[:,:,:,i],gtXYZT[:,:,:,i],4)
#        montageImg = func.createMontageRGB(volXYZT[:,:,:,i],255*bpXYZT[:,:,:,i],4)
#        im = plt.imshow(montageImg,cmap=plt.cm.gray,animated=True)
#        ims.append([im])
#    ani = animation.ArtistAnimation(fig,ims,interval=100,blit=False,repeat_delay=100)
#    plt.show()
    
    # Save raw data and ground truth in a dictionary
    rawPatientData = {}
    rawPatientData.update({'vol': volXYZT})
    rawPatientData.update({'gt': gtXYZT})
    rawPatientData.update({'bp':bpXYZT})
    rawPatientData.update({'spacing':spacing})
    rawPatientData.update({'IOP':np.asarray(saList[0][0].ImageOrientationPatient)[0:6]})
    exec("sio.savemat('N://ShusilDangi//RVSC//TrainingSetMat//rawData"+str(ind)+".mat',rawPatientData)")

#vtkImage = func.importArray(gtPh1.swapaxes(0,2).swapaxes(1,2),1,spacing)
#mesh = func.createMesh(vtkImage,255)
#
#meshActor = func.createMeshActor(mesh,(1.0,0.0,0.1),1.0)
##meshActor.GetProperty().EdgeVisibilityOn()
##meshActor1.GetProperty().LightingOff()
#
## Set Camera Position
#camera = vtkCamera()
#camera.SetViewUp(0,0,-1)
#camera.SetPosition(-2,-2,-2)
#
## Create the Renderer, Window and Interator
#ren = vtk.vtkRenderer()
#ren.SetBackground(0.1,0.1,0.2)
## Actor for isosurfaces
#ren.AddActor(meshActor)
## Set the camera
#ren.SetActiveCamera(camera)
#ren.ResetCamera()
#func.createStartWindow(ren)

# ind = 4: slice spacing non-uniform
