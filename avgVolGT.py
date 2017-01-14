# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 19:25:11 2016

@author: sxd7257
"""
import sys
sys.path.append("C:\Users\sxd7257\Dropbox\Python Scripts")
import scipy.io as sio
import myFunctions as func
import numpy as np
import SimpleITK as sitk
from vtk import *
import skimage.measure as measure
import skimage.segmentation as seg
from scipy.interpolate import griddata
from scipy.ndimage.morphology import binary_closing

matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\trainingSet.mat')
#matContents = sio.loadmat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\trainingSetMesh.mat')
keys = matContents.keys()
for i in keys:
    exec(i+"= matContents['"+i+"']")

## Histogram Matching Filter
histFilter = sitk.HistogramMatchingImageFilter()
histFilter.SetNumberOfHistogramLevels(256)
histFilter.SetNumberOfMatchPoints(5)
histFilter.SetThresholdAtMeanIntensity(True)

refVol = tformVol12
#refSpacing = np.copy(spacing)
#func.displayMontage(refVol,5)

avgVol = np.zeros(refVol.shape)
avgGT = np.zeros(refVol.shape)
#badinds = [37,41,52]

#indNG = [40,42,99]   # Patients with variable array sizes
#indNG = [8,25,37,40,42,43,47,75,92,99]  # Patients with bad mesh
indices = range(0,16)
#for i in indNG: indices.remove(i);
    
#for i in range(1,100):
for i in indices:
#    exec("vol = tformVol"+str(i))
    exec("vol = func.normalizeSlices(tformVol"+str(i)+",0,99)")
#    print(i,vol.max(),vol.min())
    exec("gt = tformGT"+str(i))
    avgVol = avgVol+vol.astype(float)/float(vol.max())
    avgGT = avgGT+gt.astype(float)/float(gt.max())
avgVol = avgVol.astype(float)/avgVol.max()*255
avgGT = avgGT.astype(float)/avgGT.max()*255

func.displayMontage(avgVol,5)
func.displayMontage(avgGT,5)
func.displayMontage(255*(avgGT>127),5)

data_matrix = 255*(avgGT>127)
#oldSpacing = spacing[0]
oldSpacing = (1.5625,1.5625,6)

basalSlice = 0
#bSlice = np.zeros((data_matrix.shape[0],data_matrix.shape[1]))
#data_matrix = np.dstack((bSlice,data_matrix,bSlice))
data_matrix = data_matrix.astype('uint8')

indNZ = np.nonzero(data_matrix)
startPosition = np.min(indNZ,axis=1)
endPosition = np.max(indNZ,axis=1)+1

endo = np.zeros(data_matrix.shape)
epi = np.zeros(data_matrix.shape)

for j in range(data_matrix.shape[2]):
    currentSlice = data_matrix[:,:,j]
    labeledImg, nLabels = measure.label(currentSlice,neighbors=8,background=0,return_num=True)
    if(nLabels>1 & j>data_matrix.shape[2]/2):
        # This should be a basal slice with disconnected chunks of myocardium
        basalSlice = j
        break            
    else:
        invImg = 255-currentSlice
        labeledImg2,nLabels2 = measure.label(invImg,neighbors=8,background=0,return_num=True)
        if(nLabels2>1):
            # This is a mid-slice with circular endo- and epi-cardium
            array,counts = np.unique(labeledImg2,return_index=False,return_inverse=False,return_counts=True)
            # Remove background
            arrayFG = array[1:]
            countsFG = counts[1:]
            # Sort labels according to area
            ind = np.argsort(countsFG)
            arrayFG = arrayFG[ind]

            # Larger connected component corresponds epi-cardium
            epiImg = np.logical_not(labeledImg2==arrayFG[-1])
            epiContour = seg.find_boundaries(epiImg)
            epi[:,:,j] = epiContour

            # Smaller connected component corresponds endo-cardium
            endoImg = (labeledImg2==arrayFG[-2])
            endoContour = seg.find_boundaries(endoImg)
            endo[:,:,j] = endoContour
                                    
        else:
            if(j<data_matrix.shape[2]/2):
            # This is a apex slice with filled epi-cardium (assuming apex is at the top)
                print('apical slice')
                epi[:,:,j] = seg.find_boundaries(labeledImg)
            else:
            # This is a basal slice with a single chunk of myocardium
                print('basal slice')
                basalSlice = j
                break


physicalEpiPoints = np.asarray(np.nonzero(epi))
physicalEpiPoints[2,:] = physicalEpiPoints[2,:]*oldSpacing[2]

physicalEndoPoints = np.asarray(np.nonzero(endo))
physicalEndoPoints[2,:] = physicalEndoPoints[2,:]*oldSpacing[2]

originalSize = np.asarray(data_matrix.shape)
newSpacing = np.copy(oldSpacing)
newSpacing[-1]=1.0
newZ = int((originalSize[2]-1)*oldSpacing[2]/newSpacing[2])+1
newSize = np.floor(np.divide(np.multiply(originalSize-1,oldSpacing),newSpacing)).astype(int)+1

gridX,gridY,gridZ = np.mgrid[0:newSize[0],0:newSize[1],0:newSize[2]]
interpEpi = griddata(physicalEpiPoints.T,np.ones(physicalEpiPoints.shape[1]),(gridX,gridY,gridZ),method='linear',fill_value=0,rescale=True)
interpEndo = griddata(physicalEndoPoints.T,np.ones(physicalEndoPoints.shape[1]),(gridX,gridY,gridZ),method='linear',fill_value=0,rescale=True)

finalVol = np.logical_xor(interpEpi>0,interpEndo>0)
    
if basalSlice:
    vol = data_matrix[:,:,basalSlice-1:]
    basalVol = func.interpBasalSlice(vol,oldSpacing,newSpacing)
    se = np.ones((3,3,3))
    mask = basalVol<=0
    mask[:,:,-int(oldSpacing[2]):]=0
    closedVol = binary_closing(mask,se)
    finalVol[:,:,-basalVol.shape[2]+1:] = closedVol[:,:,1:]

Mask = np.zeros(finalVol.shape)
Mask[startPosition[0]:endPosition[0],startPosition[1]:endPosition[1],int(startPosition[2]*oldSpacing[2]):int((endPosition[2]-1)*oldSpacing[2]+1)] = 1
finalVol = np.multiply(finalVol,Mask)

vtkImage = func.importArray(255*finalVol.swapaxes(0,2),1,newSpacing)
avgMesh = func.createFineMesh(vtkImage,isovalue=255,trgtRedDecimation=0.75,filterNoOfIterations=20,nMeshPoints=1000)
#avgMesh = func.createMesh(vtkImage,isovalue=255)

meshActor1 = func.createMeshActor(avgMesh,(1.0,0.0,0.0),1.0)
meshActor1.GetProperty().EdgeVisibilityOn()

outlineActor = func.createOutlineActor(vtkImage)

# Set Camera Position
camera = vtkCamera()
camera.SetViewUp(0,0,-1)
camera.SetPosition(-2,-2,-2)

# Create the Renderer, Window and Interator
ren = vtk.vtkRenderer()
ren.SetBackground(0.1,0.1,0.2)
# Actor for isosurfaces
ren.AddActor(meshActor1)
# Set the camera
ren.SetActiveCamera(camera)
ren.ResetCamera()

func.createStartWindow(ren)

dictVar = {}
dictVar.update({'avgVol': avgVol})
dictVar.update({'avgGT': avgGT})
dictVar.update({'spacing': spacing})
dictVar.update({'IOP':IOP})
#sio.savemat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\avgVolGTIntensity2.mat',dictVar)
sio.savemat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\avgVolGT.mat',dictVar)

#writer = vtk.vtkXMLPolyDataWriter()
#writer.SetFileName("NewMeshes\\avgMesh2.vtp")
#writer.SetInputData(avgMesh)
#writer.Write()
