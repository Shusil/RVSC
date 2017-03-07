# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:16:53 2015

@author: Shusil Dangi

"""
from vtk import *
import numpy as np
from vtk.util import numpy_support
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PyACVD import VTK_Plotting, Clustering
import skimage.segmentation as seg
import skimage.morphology as skmorph
from scipy.ndimage.morphology import distance_transform_edt
import scipy.stats as scstat
import skimage.feature as skfeat
from skimage import exposure
import cv2
import smallestCircle as sc
import math
import skimage.measure as measure
import skimage.segmentation as segment
import scipy.spatial as spatial
from skimage.draw import line
import scipy.ndimage.morphology as morph
from skimage.draw import polygon


# Function to import a python array as vtk image    
def importArray(data_matrix,noOfComponents,spacing):
    dataShape = np.array(data_matrix.shape,dtype='int')
    imageVTK = vtk.vtkImageData()
    imageVTK.SetSpacing(spacing)
    imageVTK.SetDimensions(dataShape[-1],dataShape[-2],dataShape[-3])
    imageVTK.AllocateScalars(vtk.VTK_TYPE_UINT8,noOfComponents)
    VTKdata = numpy_support.numpy_to_vtk(data_matrix.ravel(), deep=True, array_type=vtk.VTK_TYPE_UINT8)
    imageVTK.GetPointData().SetScalars(VTKdata)
    return imageVTK


def splitEndoEpi(currentSlice,endoContour=[],epiContour=[]):
    endoPts = []
    epiPts = []
    endoImg = np.zeros(currentSlice.shape)
    epiImg = np.zeros(currentSlice.shape)
    if(np.asarray(endoContour).shape[0]>0 and np.asarray(epiContour).shape[0]>0):        
        endoEpi = np.logical_or(endoContour>0,epiContour>0)
        endoEpiPts = np.asarray(np.nonzero(endoEpi))
        # Estimate the center from nearest mid-slice
        ctr = np.mean(endoEpiPts,axis=1)
    else:
        ctr = np.mean(np.asarray(np.nonzero(currentSlice)),axis=1)
    boundaryImg = seg.find_boundaries(currentSlice)
    ptsSlice = np.asarray(np.nonzero(boundaryImg))
    ptsSlice = ptsSlice-np.repeat(ctr.reshape(2,1),ptsSlice.shape[1],axis=1)
    ptsPolar = np.zeros(ptsSlice.shape)
    for k in range(ptsSlice.shape[1]):
        ptsPolar[0,k],ptsPolar[1,k] = cart2Pol(ptsSlice[0,k],ptsSlice[1,k])
    ind = np.argsort(ptsPolar[1,:])
    ptsPolar2 = ptsPolar[:,ind]
    roundAngle = np.round(ptsPolar2[1,:],1)
    unqAngle,idx,inv,counts = np.unique(roundAngle,return_index=True,return_inverse=True,return_counts=True)
    cumCount = np.cumsum(counts)
    cumCounts = np.zeros(cumCount.shape[0]+1)
    cumCounts[1:] = cumCount

    for k in range(cumCount.shape[0]):
        idxx = np.argsort(ptsPolar2[0,cumCounts[k]:cumCounts[k+1]])
        for l in range(np.floor(float(idxx.shape[0])/2).astype(int)):
            endoPts.append(ptsPolar2[:,cumCounts[k]+idxx[l]])
            epiPts.append(ptsPolar2[:,cumCounts[k]+idxx[-l-1]])

    for k in endoPts:
        ptsCart = pol2Cart(k[0],k[1])
        endoPoints = np.round(ptsCart+ctr).astype(int)
        endoImg[endoPoints[0],endoPoints[1]] = 1.0
    endoImg = skmorph.remove_small_objects(endoImg>0,min_size=3,connectivity=1,in_place=False)
    
    for k in epiPts:
        ptsCart = pol2Cart(k[0],k[1])
        epiPoints = np.round(ptsCart+ctr).astype(int)
        epiImg[epiPoints[0],epiPoints[1]] = 1.0
    epiImg = skmorph.remove_small_objects(epiImg>0,min_size=3,connectivity=1,in_place=False)

    return (1.0*endoImg,1.0*epiImg)

    
def cart2Pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)



def pol2Cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def fourierSmoothing(seg,harmonics=5):
    edgeImg = segment.find_boundaries(seg.astype(int),mode='outer')
    ind = np.nonzero(seg)
    cX,cY = np.mean(ind[0]),np.mean(ind[1])
    indEdge = np.nonzero(edgeImg)
    r,phi = cart2Pol(indEdge[0]-cX,indEdge[1]-cY)
    indSort = np.argsort(phi)
    r = r[indSort]
    phi = phi[indSort]
    rft = np.fft.rfft(r)
    rft[harmonics:]=0
    r_smooth = np.fft.irfft(rft)
    if(len(r_smooth)<len(phi)):
        r_smooth = np.append(r_smooth,r_smooth[-1])
    x_smooth,y_smooth = pol2Cart(r_smooth,phi)
    xs,ys = x_smooth+cX,y_smooth+cY
    rr,cc = polygon(np.round(ys),np.round(xs))
    smoothImg = np.zeros(seg.shape)
    smoothImg[cc,rr] = 1
    return smoothImg
    
  
  
def getRegion(myoBinary,option='min'):
    labeledImg,nLabels = measure.label(myoBinary,background=0,return_num=True,connectivity=myoBinary.ndim)
    if(nLabels>1):
        props = measure.regionprops(labeledImg)
        area = []
        for j in props:
            area.append(j.area)
        if(option=='min'):
            Ind = np.argmin(np.asarray(area))
        else:
            Ind = np.argmax(np.asarray(area))
        endo = np.zeros(labeledImg.shape)
        endo[labeledImg==(Ind+1)] = 1
    else:
         endo = np.zeros(labeledImg.shape)
         endo[labeledImg==-1] = 1
    return(endo,nLabels)


def findBPSegment(seg,bpMask):
    # Find the connected component closest to the center of bpMask
    bpCtr = np.mean(np.asarray(np.nonzero(bpMask)),axis=1).astype(float)
    labeledImg,nLabels = measure.label(seg,background=0,return_num=True,connectivity=1)
    props = measure.regionprops(labeledImg)
    distCtr = []
    for k in range(len(props)):
        labelImg = np.zeros(labeledImg.shape)
        labelImg[labeledImg==(k+1)]=1
        distMapLabel = distance_transform_edt(np.logical_not(labelImg),sampling=(1,1),return_distances=True,return_indices=False)
        distCtr.append(distMapLabel[int(bpCtr[0]),int(bpCtr[1])])
    segBP = np.zeros(seg.shape)
    if(len(distCtr)>0):
        indexC = np.argmin(np.asarray(distCtr))
        segBP[labeledImg==(indexC+1)]=255
    return segBP



def fitEllipse(segBP):
    # Spline try
    points = np.asarray(np.nonzero(segment.find_boundaries(segBP.astype(int),connectivity=2,mode='inner',background=0))).T
    hull = spatial.ConvexHull(points)
    x = points[hull.vertices,0]
    y = points[hull.vertices,1]
    ellipse = measure.EllipseModel()
    ellipse.estimate(np.asarray([x,y]).T)
    angles = np.radians(np.arange(0,362,0.5))
    xy = np.round(np.asarray(ellipse.predict_xy(angles))).astype(int)
    img = np.zeros(segBP.shape)
    for i in range(xy.shape[0]-1):
        rr,cc = line(xy[i,0],xy[i,1],xy[i+1,0],xy[i+1,1])
        rr[rr<0] = 0
        rr[rr>=img.shape[0]] = img.shape[0]-1
        cc[cc<0] = 0
        cc[cc>=img.shape[1]] = img.shape[1]-1
        img[rr,cc]=1
    seg = morph.binary_fill_holes(img)
    return(seg)
    
    

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)
        
        
def angle_between_vectors(v0, v1, directed=True, axis=0):
    """Return angle between vectors.

    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.

    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    >>> numpy.allclose(a, math.pi)
    True
    >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    >>> numpy.allclose(a, 0)
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> a = angle_between_vectors(v0, v1)
    >>> numpy.allclose(a, [0, 1.5708, 1.5708, 0.95532])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> a = angle_between_vectors(v0, v1, axis=1)
    >>> numpy.allclose(a, [1.5708, 1.5708, 1.5708, 0.95532])
    True

    """
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    dot = np.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return np.arccos(dot if directed else np.fabs(dot))


def findRotation(IOPMoving,IOPFixed):
    axisMoving = np.cross(IOPMoving[0:3],IOPMoving[3:6])
    angle = angle_between_vectors(IOPFixed[0:3],IOPMoving[0:3],directed=False)
    rotation1 = sitk.VersorTransform(tuple(axisMoving),angle)
    rotation2 = sitk.VersorTransform(tuple(axisMoving),-angle)
    tPoint1 = rotation1.TransformPoint(IOPFixed[0:3])
    tPoint2 = rotation2.TransformPoint(IOPFixed[0:3])
    if(np.sum(np.abs(np.asarray(tPoint1)-IOPMoving[0:3]))<np.sum(np.abs(np.asarray(tPoint2)-IOPMoving[0:3]))):
        rotation = sitk.VersorTransform((0,0,1),angle)
    else:
        rotation = sitk.VersorTransform((0,0,1),-angle)
    return rotation

#def importArray2(data_matrix,spacing):
#    data_matrix = fixArray(data_matrix)
#    dataShape = np.array(data_matrix.shape,dtype='int')
#    imageVTK = vtk.vtkImageData()
#    imageVTK.SetSpacing(spacing)
#    imageVTK.SetDimensions(dataShape[3],dataShape[2],dataShape[1])
#    imageVTK.AllocateScalars(vtk.VTK_TYPE_UINT8,3)
#    VTKdata = numpy_support.numpy_to_vtk(data_matrix.ravel(), deep=True, array_type=vtk.VTK_TYPE_UINT8)
#    imageVTK.GetPointData().SetScalars(VTKdata)
#    return imageVTK
    

# Function to resample the volume to new voxel spacing
def resampleVolume(inputVol,NewVoxelSpacing,interpolator):
    originalSize = np.asarray(inputVol.GetSize())
    originalSpacing = np.asarray(inputVol.GetSpacing())
    newSize = np.floor(np.divide(np.multiply(originalSize-1,originalSpacing),NewVoxelSpacing)).astype(int)+1
    resampleVolumeFilter = sitk.ResampleImageFilter()
    resampleVolumeFilter.SetOutputDirection(inputVol.GetDirection())
    resampleVolumeFilter.SetOutputOrigin(inputVol.GetOrigin())
    resampleVolumeFilter.SetOutputPixelType(inputVol.GetPixelIDValue())
    resampleVolumeFilter.SetOutputSpacing(NewVoxelSpacing)
    resampleVolumeFilter.SetDefaultPixelValue(-10)
    resampleVolumeFilter.SetTransform(sitk.Transform(3,sitk.sitkIdentity))
    resampleVolumeFilter.SetSize(newSize)
    # resampleVolumeFilter.SetExtrapolator()
    if interpolator=='nn':
        resampleVolumeFilter.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolator=='g':
        resampleVolumeFilter.SetInterpolator(sitk.sitkGaussian)
    elif interpolator=='b':
        resampleVolumeFilter.SetInterpolator(sitk.sitkBSpline)
    else:
        resampleVolumeFilter.SetInterpolator(sitk.sitkLinear)

    resampledVol = resampleVolumeFilter.Execute(inputVol)
    return resampledVol



# def stretchContrast(img,lower_clipping_percent,higher_clipping_percent):
#    outImg = np.zeros(img.shape)
#    img = (img.astype(float)/img.max()*255).astype(int)
#    if(img.min()<img.max()):
#        t = np.zeros(256)
#        hist = np.histogram(img,bins=256,range=(0,256),density=True)
#        cumHist = np.cumsum(hist[0])
#        x = cumHist<lower_clipping_percent
#        ind = np.nonzero(x)
#        if(len(ind[0])==0):
#            fmin = 0
#        else:
#            fmin = ind[0][-1]
#        y = cumHist<higher_clipping_percent
#        ind = np.nonzero(y)
#        fmax = ind[0][-1]
#        for k in range(0,255):
#            t[k] = round(float(k-fmin)/(fmax-fmin)*255)
#            if(t[k]<0): t[k]=0
#            if(t[k]>255): t[k]=255
#
#        for m in range(img.shape[0]):
#            for n in range(img.shape[1]):
#                outImg[m,n] = t[img[m,n]]
#    return outImg


    
# def appendSlices(inputVol,noOfSlices):
#    appImg = np.zeros((volSize[0],volSize[1],volSize[2]+2*noOfSlices))
#    appImg[:,:,0:noOfSlices] = npImage[:,:,0]
#    appImg[:,:,noOfSlices:-noOfSlices] = npImage
#    appImg[:,:,-1] = npImage[:,:,-1]


# # Function to resample the volume to new voxel spacing
# def resampleVolume(inputVolRaw,NewVoxelSpacing,interpolator):
#
#    volSize = np.asarray(inputVolRaw.GetSize())
#    volSpacing = np.asarray(inputVolRaw.GetSpacing())
# #    if(len(newSize)!=0):
# #        newVolSize = newSize
# #    else:
#    newVolSize = np.floor(np.divide(np.multiply(volSize-1,volSpacing),NewVoxelSpacing)).astype(int)+1
# #    sizeZ = float(newVolSize[2])
#
#    # duplicate first and last slices
#    npImage = sitk.GetArrayFromImage(inputVolRaw).swapaxes(0,2)
#    appImg = np.zeros((volSize[0],volSize[1],volSize[2]+2))
#    appImg[:,:,0] = npImage[:,:,0]
#    appImg[:,:,1:-1] = npImage
#    appImg[:,:,-1] = npImage[:,:,-1]
#
#    # create new sitk image with duplicated first and last slices
#    inputVol = sitk.GetImageFromArray(appImg.swapaxes(0,2))
#    inputVol.SetSpacing(volSpacing)
#    inputVol.SetOrigin(inputVolRaw.GetOrigin())
#    inputVol = sitk.Cast(inputVol,inputVolRaw.GetPixelID())
#
#    originalSize = np.asarray(inputVol.GetSize())
#    originalSpacing = np.asarray(inputVol.GetSpacing())
#    newSize = np.floor(np.divide(np.multiply(originalSize,originalSpacing),NewVoxelSpacing)).astype(int)
#
#    resampleVolumeFilter = sitk.ResampleImageFilter()
#    resampleVolumeFilter.SetOutputDirection(inputVol.GetDirection())
#    resampleVolumeFilter.SetOutputOrigin(inputVol.GetOrigin())
#    resampleVolumeFilter.SetOutputPixelType(inputVol.GetPixelIDValue())
#    resampleVolumeFilter.SetOutputSpacing(NewVoxelSpacing)
#    resampleVolumeFilter.SetDefaultPixelValue(float('nan'))
#    resampleVolumeFilter.SetTransform(sitk.Transform(3,sitk.sitkIdentity))
#    resampleVolumeFilter.SetSize(newSize)
# #    resampleVolumeFilter.SetExtrapolator()
#    if interpolator=='nn':
#        resampleVolumeFilter.SetInterpolator(sitk.sitkNearestNeighbor)
#    elif interpolator=='g':
#        resampleVolumeFilter.SetInterpolator(sitk.sitkGaussian)
#    elif interpolator=='b':
#        resampleVolumeFilter.SetInterpolator(sitk.sitkBSpline)
#    else:
#        resampleVolumeFilter.SetInterpolator(sitk.sitkLinear)
#
#    resampledVol = resampleVolumeFilter.Execute(inputVol)
#
# #    npRSImage = sitk.GetArrayFromImage(resampledVol).swapaxes(0,2)
# #    ctrSlice = np.floor(float(npRSImage.shape[2])/2) - np.floor(float(volSpacing[2])/(2*NewVoxelSpacing[2]))
# #    startSlice = max(ctrSlice-np.ceil(sizeZ/2),0)
# #    endSlice = startSlice+sizeZ
# #    ctrVol = npRSImage[:,:,startSlice:endSlice]
# #
# #    rsVol = sitk.GetImageFromArray(ctrVol.swapaxes(0,2))
# #    rsVol.SetSpacing(originalSpacing)
# #    rsVol.SetOrigin(inputVol.GetOrigin())
# #    rsVol = sitk.Cast(rsVol,inputVol.GetPixelID())
#
#    return resampledVol
#
#
# Attempt to resample reference patient to same number of slices for patient in hand
# def resampleRefVol(sitkRefVol,nSlices):
#    oldSize = sitkRefVol.GetSize()
#    newSize = oldSize[0:2]+(nSlices,)
#    oldVoxelSpacing = sitkRefVol.GetSpacing()
#    zSpacing = oldVoxelSpacing[2]*oldSize[2]/nSlices
#    newVoxelSpacing = oldVoxelSpacing[0:2]+(zSpacing,)
#    resampleVolumeFilter = sitk.ResampleImageFilter()
#    sitkRefVolRS = resampleVolumeFilter.Execute(sitkRefVol,newSize,sitk.Transform(),sitk.sitkNearestNeighbor,sitkRefVol.GetOrigin(),newVoxelSpacing,sitkRefVol.GetDirection(),0,sitkRefVol.GetPixelIDValue())
#    return sitkRefVolRS
#
#
# Attempt to shift the physical space of volume to be resampled
# def resampleVolume(inputVol,NewVoxelSpacing,interpolator):
#    originalSize = np.asarray(inputVol.GetSize())
#    originalSpacing = np.asarray(inputVol.GetSpacing())
#    newSize = np.round(np.divide(np.multiply(originalSize,originalSpacing),NewVoxelSpacing)).astype(int)
#    if(round(originalSize[2]*originalSpacing[2]/NewVoxelSpacing[2])!=(originalSize[2]*originalSpacing[2]/NewVoxelSpacing[2])):
#        _,origin = divmod(originalSize[2]*originalSpacing[2],NewVoxelSpacing[2])
#        origin = float(origin)/NewVoxelSpacing[2]
#        inputOrigin = inputVol.GetOrigin();
#        newOrigin = inputOrigin[0:2]+(-origin/2,)
#        inputVol.SetOrigin(newOrigin)
#
#    resampleVolumeFilter = sitk.ResampleImageFilter()
#    if interpolator=='nn':
#        resampledVol = resampleVolumeFilter.Execute(inputVol,newSize,sitk.Transform(),sitk.sitkNearestNeighbor,(0.0,0.0,0.0),NewVoxelSpacing,inputVol.GetDirection(),0,inputVol.GetPixelIDValue())
#    if interpolator=='l':
#        resampledVol = resampleVolumeFilter.Execute(inputVol,newSize,sitk.Transform(),sitk.sitkLinear,(0.0,0.0,0.0),NewVoxelSpacing,inputVol.GetDirection(),0,inputVol.GetPixelIDValue())
#    return resampledVol


def createMontage(vol,nCols=4):
    if(len(vol.shape)>2):
        nSlices = vol.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImg = 128*np.ones((nRows*vol.shape[0],nCols*vol.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImg[i*vol.shape[0]:(i+1)*vol.shape[0],j*vol.shape[1]:(j+1)*vol.shape[1]] = vol[:,:,i*nCols+j]
    return montageImg


def createMontageRGB(vol_fixed, vol_moving, nCols=4, expand=1):
    if(len(vol_fixed.shape)>2):
        nSlices = vol_fixed.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImgFixed = 128*np.ones((nRows*vol_fixed.shape[0],nCols*vol_fixed.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImgFixed[i*vol_fixed.shape[0]:(i+1)*vol_fixed.shape[0],j*vol_fixed.shape[1]:(j+1)*vol_fixed.shape[1]] = vol_fixed[:,:,i*nCols+j]
    
        nSlices = vol_moving.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImgMoving = 128*np.ones((nRows*vol_moving.shape[0],nCols*vol_moving.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImgMoving[i*vol_moving.shape[0]:(i+1)*vol_moving.shape[0],j*vol_moving.shape[1]:(j+1)*vol_moving.shape[1]] = vol_moving[:,:,i*nCols+j]
    else:
        montageImgFixed = vol_fixed
        montageImgMoving = vol_moving

    simg1 = sitk.GetImageFromArray(montageImgFixed)
    simg2 = sitk.GetImageFromArray(montageImgMoving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(simg1),sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(simg2),sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg1/2.+simg2/2., simg2)
    if expand != 1:
        cimg = sitk.Expand(cimg, [expand]*3)
    del simg1, simg2

    aimg = sitk.GetArrayFromImage(cimg)
    return(aimg)


# Function that displays the volume as montage image with 4 columns
def displayMontage(vol,nCols=1):
    if(len(vol.shape)>2):
        nSlices = vol.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImg = 128*np.ones((nRows*vol.shape[0],nCols*vol.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImg[i*vol.shape[0]:(i+1)*vol.shape[0],j*vol.shape[1]:(j+1)*vol.shape[1]] = vol[:,:,i*nCols+j]
        plt.figure(),plt.imshow(montageImg,cmap=plt.cm.gray)
    else:
        plt.figure(),plt.imshow(vol,cmap=plt.cm.gray)



def displayMontageRGB(vol_fixed, vol_moving, nCols=1, title="", margin=0.05, expand=1):
    if(len(vol_fixed.shape)>2):
        nSlices = vol_fixed.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImgFixed = 128*np.ones((nRows*vol_fixed.shape[0],nCols*vol_fixed.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImgFixed[i*vol_fixed.shape[0]:(i+1)*vol_fixed.shape[0],j*vol_fixed.shape[1]:(j+1)*vol_fixed.shape[1]] = vol_fixed[:,:,i*nCols+j]
    
        nSlices = vol_moving.shape[2]
        nRows = np.ceil(float(nSlices)/nCols).astype(int)
        montageImgMoving = 128*np.ones((nRows*vol_moving.shape[0],nCols*vol_moving.shape[1]))
        for i in range(nRows):
            for j in range(nCols):
                if((i*nCols+j)<nSlices):
                    montageImgMoving[i*vol_moving.shape[0]:(i+1)*vol_moving.shape[0],j*vol_moving.shape[1]:(j+1)*vol_moving.shape[1]] = vol_moving[:,:,i*nCols+j]
    else:
        montageImgFixed = vol_fixed
        montageImgMoving = vol_moving

    simg1 = sitk.GetImageFromArray(montageImgFixed)
    simg2 = sitk.GetImageFromArray(montageImgMoving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(simg1), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(simg2), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg1/2.+simg2/2., simg2)
    if expand != 1:
        cimg = sitk.Expand(cimg, [expand]*3)
    del simg1, simg2

    aimg = sitk.GetArrayFromImage(cimg)
    
    if aimg.ndim == 3:
        ysize,xsize,c = aimg.shape
        if not c in (3,4):
            aimg = aimg[c//2,:,:]
    else:
        ysize,xsize = aimg.shape
        
    dpi=80
    # Make a figure big enough to accomodate an axis of xpixels by ypixels
    # as well as the ticklabels, etc...
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Make the axis the right size...
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    
    extent = (0,xsize*cimg.GetSpacing()[1],0,ysize*cimg.GetSpacing()[0])
    t = ax.imshow(aimg,extent=extent)
    
    if aimg.ndim == 2:
        t.set_cmap("gray")
    
    if(title):
        plt.title(title)



# Function to fix the axis for sitk vs numpy array
def fixArray(img):
    l = len(img.shape)
    for i in range(0,int(np.ceil(l/2))):
        img = img.swapaxes(i,l-i-1)
    return(img)



# Function that normalizes (0-1) the input volume slice by slice
def normalizeSlices(initVol,lowerPercentile=0,higherPercentile=100,mask=[]):
    finalVol = np.zeros(initVol.shape).astype('uint8')
    if(len(initVol.shape)==3):
        for i in range(initVol.shape[2]):
            if(initVol[:,:,i].max()>0):
                temp = stretchContrast(initVol[:,:,i],lowerPercentile,higherPercentile,mask)
                finalVol[:,:,i] = (255*(temp.astype('float')-float(temp.min()))/float(temp.max())).astype('uint8')
                finalVol[:,:,i] = exposure.rescale_intensity(finalVol[:,:,i])
    if(len(initVol.shape)==2):
        if(initVol.max()>0):
            temp = stretchContrast(initVol,lowerPercentile,higherPercentile,mask)
            finalVol = (255*(temp.astype('float')-float(temp.min()))/float(temp.max())).astype('uint8')
            finalVol = exposure.rescale_intensity(finalVol)
    return finalVol



def stretchContrast(img,lowerPercentile=0,higherPercentile=100,mask=[]):
    if(np.sum(mask)>0):
        indNZ = np.nonzero(mask)
    else:
        indNZ = np.nonzero(np.ones(img.shape))
    nImg = np.copy(img)
    vMin,vMax = np.percentile(img[indNZ[0],indNZ[1]].ravel(),(lowerPercentile,higherPercentile))
    nImg[nImg>vMax] = vMax
    nImg[nImg<vMin] = vMin
    return(nImg)



def getCircularMask(mask,shape,scale=1.15,offset=False):
    points = np.asarray(np.nonzero(mask))
    points = tuple(points.T)
    circ = sc.make_circle(points)
    circ = np.round(circ).astype(int)
    img = np.zeros(mask.shape).astype('uint8')
    if(offset==True):
        perturb = int(round(float(2*circ[2])*np.random.rand(1)[0]-float(circ[2])))
    else:
        perturb = 0
    cv2.circle(img, center=(circ[1]+perturb, circ[0]+perturb), radius=int(circ[2]*scale), color=1, thickness=-1)
    maskVol = np.zeros(shape)
    if(len(maskVol.shape)>2):
        for i in range(maskVol.shape[2]):
            maskVol[:,:,i]=img
    else:
        maskVol=img
    return (maskVol,circ[2])
    

def findEvaluationMask(refGT):
    refGT = refGT.astype(float)/np.max(refGT)
    refGT = refGT>=0.5
    mask = np.zeros(refGT.shape)
    if(np.sum(refGT.ravel())>0):
        props = measure.regionprops(refGT)
        r = props[0].equivalent_diameter/8
        mask = skmorph.binary_dilation(refGT,selem=skmorph.disk(r))
    return(mask)
    

    
def evalMetrics(refGT,segGT,metricMask):
    metricMask = metricMask>0    
    
    refGT = refGT.astype(float)/np.max(refGT)
    refGT = refGT>=0.5

    segGT = segGT.astype(float)/np.max(segGT)
    segGT = segGT>=0.5

    T1 = float(np.sum(np.logical_and(refGT,segGT))) #true positive
    F1 = float(np.sum(np.logical_and(np.logical_and(np.logical_not(refGT),segGT),metricMask))) #false positive
    T0 = float(np.sum(np.logical_and(np.logical_and(np.logical_not(refGT),np.logical_not(segGT)),metricMask))) #true negative
    F0 = float(np.sum(np.logical_and(np.logical_and(refGT,np.logical_not(segGT)),metricMask))) #false negative

    sensitivity = 0
    specificity = 0
    positivePredictiveValue = 0
    negativePredictiveValue = 0
    Dice = 0
    Jaccard = 0
    
    if((T1+F0)>0):
        sensitivity = T1/(T1+F0)

    if((T0+F1)>0):
        specificity = T0/(T0+F1)
        
    if((T1+F1)>0):
        positivePredictiveValue = T1/(T1+F1)
        
    if((T0+F0)>0):
        negativePredictiveValue = T0/(T0+F0)
    
    A1 = float(np.sum(refGT.ravel()))
    A2 = float(np.sum(segGT.ravel()))
    union = float(np.sum(np.logical_or(refGT,segGT).ravel()))
    
    if((A1+A2)>0):
        Dice = 2*T1/(A1+A2)
    if((union)>0):
        Jaccard = T1/union
    
    return(sensitivity,specificity,positivePredictiveValue,negativePredictiveValue,Dice,Jaccard)



def evaluateMetrics(testGT,tformGT,option='intersection'):
    dice = ()
    jaccard = ()
    sensitivity = ()
    specificity = ()
    PPV = ()
    NPV = ()
    
    testGT = testGT/testGT.max()
    tformGT = tformGT/tformGT.max()

    indNZTest = np.nonzero(testGT)
    startInd1 = np.min(indNZTest,axis=1)
    endInd1 = np.max(indNZTest,axis=1)

    indNZTform = np.nonzero(tformGT)
    startInd2 = np.min(indNZTform,axis=1)
    endInd2 = np.max(indNZTform,axis=1)

    if(option=='union'):
        startInd = min(startInd1[2],startInd2[2])
        endInd = max(endInd1[2],endInd2[2])
    if(option=='intersection'):
        startInd = max(startInd1[2],startInd2[2])
        endInd = min(endInd1[2],endInd2[2])
    if(option=='reference'):
        startInd = startInd1[2]
        endInd = endInd1[2]
    
    for k in range(startInd,endInd+1):
        maskRefGT = findEvaluationMask(testGT[:,:,k]>0.5)
        maskSegGT = tformGT[:,:,k]>0.5
        mask = np.logical_or(maskRefGT,maskSegGT)
        if(np.sum(mask)>0):
            a,b,c,d,e,f = evalMetrics(testGT[:,:,k],tformGT[:,:,k],maskRefGT)
            sensitivity = sensitivity+(a,)
            specificity = specificity+(b,)
            PPV = PPV+(c,)
            NPV = NPV+(d,)
            dice = dice+(e,)
            jaccard = jaccard+(f,)
    return(dice,jaccard,sensitivity,specificity,PPV,NPV)

    
# Histogram match srcVol to tmpltVol slice by slice
#def histMatch(srcVol, tmpltVol):
#    """
#    Adjust the pixel values of a grayscale image such that its histogram
#    matches that of a target image
#
#    Arguments:
#    -----------
#        srcVol: np.ndarray
#            Image to transform; the histogram is computed over the flattened
#            array
#        tmpltVol: np.ndarray
#            Template image; can have different dimensions to srcVol
#    Returns:
#    -----------
#        matched: np.ndarray
#            The transformed output image
#    """
#
#    oldShape = srcVol.shape
#    eqVol = srcVol.copy()
#
#    for i in range(oldShape[2]):
#        # get the set of unique pixel values and their corresponding indices and
#        # counts
#        s_values, bin_idx, s_counts = np.unique(srcVol[:,:,i].ravel(), return_inverse=True,return_counts=True)
#        t_values, t_counts = np.unique(tmpltVol[:,:,i].ravel(), return_counts=True)
#    
#        # take the cumsum of the counts and normalize by the number of pixels to
#        # get the empirical cumulative distribution functions for the source and
#        # tmpltVol images (maps pixel value --> quantile)
#        s_quantiles = np.cumsum(s_counts).astype(np.float64)
#        s_quantiles /= s_quantiles[-1]
#        t_quantiles = np.cumsum(t_counts).astype(np.float64)
#        t_quantiles /= t_quantiles[-1]
#    
#        # interpolate linearly to find the pixel values in the template image
#        # that correspond most closely to the quantiles in the source image
#        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
#
#        eqVol[:,:,i] = interp_t_values[bin_idx].reshape(oldShape[0],oldShape[1])
#    return eqVol


## Function that matches histogram of srcVol to that of tmpltVol
#def histMatch(srcVol, tmpltVol):
#    """
#    Adjust the pixel values of a grayscale image such that its histogram
#    matches that of a target image
#
#    Arguments:
#    -----------
#        srcVol: np.ndarray
#            Image to transform; the histogram is computed over the flattened
#            array
#        tmpltVol: np.ndarray
#            Template image; can have different dimensions to srcVol
#    Returns:
#    -----------
#        matched: np.ndarray
#            The transformed output image
#    """
#
#    oldShape = srcVol.shape
#    eqVol = srcVol.copy()
#
#    # get the set of unique pixel values and their corresponding indices and
#    # counts
#    s_values, bin_idx, s_counts = np.unique(srcVol.ravel(), return_inverse=True,return_counts=True)
#    t_values, t_counts = np.unique(tmpltVol.ravel(), return_counts=True)
#
#    # take the cumsum of the counts and normalize by the number of pixels to
#    # get the empirical cumulative distribution functions for the source and
#    # tmpltVol images (maps pixel value --> quantile)
#    s_quantiles = np.cumsum(s_counts).astype(np.float64)
#    s_quantiles /= s_quantiles[-1]
#    t_quantiles = np.cumsum(t_counts).astype(np.float64)
#    t_quantiles /= t_quantiles[-1]
#
#    # interpolate linearly to find the pixel values in the template image
#    # that correspond most closely to the quantiles in the source image
#    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
#
#    eqVol = interp_t_values[bin_idx].reshape(oldShape)
#    return (eqVol,interp_t_values)


def hist_match(source, template, sourceMask=[], templateMask=[]):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    idxSrc = np.nonzero(sourceMask)
    srcFlat = source[idxSrc[0],idxSrc[1]].ravel()
    idxTmplt = np.nonzero(templateMask)
    tmpltFlat = template[idxTmplt[0],idxTmplt[1]].ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    t_values, t_counts = np.unique(tmpltFlat, return_counts=True)

    s_counts,s_values = np.histogram(srcFlat,bins=256,range=(0,256))
    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    if(np.sum(np.isnan(interp_t_values))>0):
        print(interp_t_values)
        print('problem')

    return interp_t_values[source.astype(int).ravel()].reshape(oldshape)
    
    
    
# Function to import MATLAB .mat files    
def loadMatFile(Filename):
    fullFilename = Filename+'.mat'
    matContents = sio.loadmat(fullFilename)
    arr = matContents[Filename]
    return arr


def createMesh(vtkData,isovalue):
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkData)
    contour.SetValue(0,isovalue)
    contour.Update()
    return contour.GetOutput()


def createDeciMesh(vtkData,isovalue,trgtReduction):
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkData)
    contour.SetValue(0,isovalue)
    contour.Update()
    
    deci = vtk.vtkDecimatePro()
    deci.SetInputConnection(contour.GetOutputPort())
    deci.SetTargetReduction(trgtReduction)
    deci.PreserveTopologyOn()
    # deci.SetFeatureAngle(30)
    deci.SplittingOn()
    deci.Update()
    
    return deci.GetOutput()

def createSmoothMesh(vtkData,isovalue,trgtReduction,filterNoOfIterations):
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkData)
    contour.SetValue(0,isovalue)
    contour.Update()
    
    deci = vtk.vtkDecimatePro()
    deci.SetInputConnection(contour.GetOutputPort())
    deci.SetTargetReduction(trgtReduction)
    deci.PreserveTopologyOn()
    # deci.SetFeatureAngle(30)
    deci.SplittingOn()
    deci.Update()
    
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(contour.GetOutputPort())
    smoother.SetNumberOfIterations(filterNoOfIterations)
    # smoother.FeatureEdgeSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.BoundarySmoothingOff()
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    return smoother.GetOutput()

def createMeshActor(mesh, color, opacity):
    meshNormals = vtk.vtkPolyDataNormals()
    meshNormals.SetInputData(mesh)
    meshNormals.SetFeatureAngle(60.0)
    
    meshStripper = vtk.vtkStripper()
    meshStripper.SetInputConnection(meshNormals.GetOutputPort())
    
    meshMapper = vtk.vtkPolyDataMapper()
    meshMapper.SetInputConnection(meshStripper.GetOutputPort())
    meshMapper.ScalarVisibilityOff()
    
    meshProperty = vtk.vtkProperty()
    meshProperty.SetColor(color)
    meshProperty.SetOpacity(opacity)
    
    meshActor = vtk.vtkActor()
    meshActor.SetMapper(meshMapper)
    meshActor.SetProperty(meshProperty)
    
    return meshActor



def createOutlineActor(vtkData):
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputData(vtkData)
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetColor(0.9,0.9,0.9)
    return outlineActor



def createStartWindow(ren):
    # Create the window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName("Volume rendering")
    renWin.SetSize(1000,1000)
    
    # Start the application
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.Initialize()
    iren.Start()
    
    return()



def visualizeCutPlane(imageVTK):
    global sliceNum, renWin, d
    sliceNum=0
    w,h,d = imageVTK.GetDimensions()
    # # Create the MIP Volume Renderer
    # MIP = vtk.vtkVolumeRayCastMIPFunction()
    # volumeMapper = vtk.vtkVolumeRayCastMapper()
    # volumeMapper.SetVolumeRayCastFunction(MIP)
    # volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    #
    # # Create a Volume
    # volume = vtk.vtkVolume()
    # volume.SetMapper(volumeMapper)
    
    # Create an outline
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputData(imageVTK)
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outlineMapper)
    outline.GetProperty().SetColor(0.9,0.9,0.9)
    
    # Initialize a Plane
    plane = vtk.vtkImagePlaneWidget()
    plane.SetInputData(imageVTK)
    plane.SetSliceIndex(sliceNum)
    
    # Set Camera Position
    camera = vtk.vtkCamera()
    camera.SetViewUp(0,0,-1)
    camera.SetPosition(-2,-2,-2)
    
    # Display the slice number
    textActor = vtk.vtkTextActor()
    tp = vtk.vtkTextProperty()
    tp.SetColor(1.0,0.2,0.3)
    tp.SetFontSize(30)
    textActor.SetTextProperty(tp)
    textActor.SetInput(str(sliceNum))
    
    # Create the Renderer, Window and Interator
    ren = vtk.vtkRenderer()
    #ren.AddVolume(volume)
    ren.SetBackground(0.1,0.1,0.2)
    # Actor for outline
    ren.AddActor(outline)
    # Set the camera
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    
    # Create the window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName("Volume rendering")
    renWin.SetSize(1000,1000)
    
    
    # Detect Keypress events
    def Keypress(obj, event):
        global sliceNum, renWin
        key = obj.GetKeySym()
        if key == "Left":
            sliceNum = max(sliceNum-1,0)
            plane.SetSliceIndex(sliceNum)
            textActor.SetInput(str(sliceNum))
            renWin.Render()
        elif key == "Right":
            sliceNum = min(sliceNum+1,w)
            plane.SetSliceIndex(sliceNum)
            textActor.SetInput(str(sliceNum))
            renWin.Render()
            
    # Start the application
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    # Re-render according to the keypress
    iren.AddObserver("KeyPressEvent",Keypress)
    
    # For cut plane
    plane.SetPlaneOrientationToXAxes()
    # plane.SetPlaneOrientationToYAxes()
    # plane.SetPlaneOrientationToZAxes()
    # volume.VisibilityOff()
    plane.SetInteractor(iren)
    plane.EnabledOn()
    
    iren.Initialize()
    iren.Start()



# Function to display Maximum Intensity Projection
def visualizeMIP(data_matrix,spacing):
    dataImporter = importArray(data_matrix,1,spacing)

    # Create the MIP Volume Renderer
    MIP = vtk.vtkVolumeRayCastMIPFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(MIP)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
    
    # Create a Volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    
    # Create an outline
    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputConnection(dataImporter.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outlineMapper)
    outline.GetProperty().SetColor(0.9,0.9,0.9)
    
    # # Initialize a Plane
    # plane = vtkImagePlaneWidget()
    # plane.SetInputConnection(dataImporter.GetOutputPort())
    # plane.SetSliceIndex(3)
    
    # Set Camera Position
    camera = vtk.vtkCamera()
    camera.SetViewUp(0,0,-1)
    camera.SetPosition(-2,-2,-2)
    
    # Create the Renderer, Window and Interator
    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    ren.SetBackground(0.1,0.1,0.2)
    # Actor for outline
    ren.AddActor(outline)
    # Set the camera
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    
    # Create the window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName("Volume rendering")
    renWin.SetSize(1000,1000)
    
    # Start the application
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # # For cut plane
    # plane.SetPlaneOrientationToXAxes()
    # #plane.SetPlaneOrientationToYAxes()
    # #plane.SetPlaneOrientationToZAxes()
    # volume.VisibilityOff()
    # plane.SetInteractor(iren)
    # plane.EnabledOn()
    
    iren.Initialize()
    iren.Start()
    return


# Function that performs Composite Volume Rendering
def visualizeCompositeVol(vtkImage):
    # vtkImage = importArray(data_matrix,1,spacing)
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    colorFunc = vtk.vtkColorTransferFunction()
    maxI = vtkImage.GetScalarTypeMax()
    for i in range(int(maxI)+1):
        alphaChannelFunc.AddPoint(i, 0.1)
        intensity = i/float(maxI)
        colorFunc.AddRGBPoint(i,intensity,intensity,intensity)
    
    # Set black opacity to 0 to see the sample
    alphaChannelFunc.AddPoint(0,0.0)
    
    # The preavius two classes stored properties. Because we want to apply these
    # properties to the volume we want to render, we have to store them in a class 
    # that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    # volumeProperty.ShadeOn()
    
    # This class describes how the volume is rendered (through ray tracing).
    # For Composite Volume Rendering
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    
    # function to reduce the spacing between each image
    # volumeMapper.SetMaximumImageSampleDistance(0.01)
    
    # We can finally create our volume. We also have to specify the data for it, 
    # as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    # volumeMapper.SetInputConnection(smooth.GetOutputPort())
    # volumeMapper.SetInputData(smooth)
    volumeMapper.SetInputData(vtkImage)
    
    # The class vtkVolume is used to pair the previously declared volume as well 
    # as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
    # Create an outline
    outlineData = vtk.vtkOutlineFilter()
    # outlineData.SetInputData(smooth)
    outlineData.SetInputData(vtkImage)
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(outlineMapper)
    outline.GetProperty().SetColor(0.9,0.9,0.9)
    
    # Set Camera Position
    camera = vtk.vtkCamera()
    camera.SetViewUp(0,0,-1)
    camera.SetPosition(-2,-2,-2)
    
    # # Initialize a Plane
    # plane = vtkImagePlaneWidget()
    # plane.SetInputConnection(dataImporter.GetOutputPort())
    # plane.SetSliceIndex(3)
    
    # Create the Renderer, Window and Interator
    ren = vtk.vtkRenderer()
    ren.AddVolume(volume)
    ren.SetBackground(0.1,0.1,0.2)
    # Actor for outline
    ren.AddActor(outline)
    # Set the camera
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    
    # Create the window
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName("Volume rendering")
    renWin.SetSize(1000,1000)
    
    # Start the application
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # For cut plane
    # plane.SetPlaneOrientationToXAxes()
    # #plane.SetPlaneOrientationToYAxes()
    # #plane.SetPlaneOrientationToZAxes()
    # volume.VisibilityOff()
    # plane.SetInteractor(iren)
    # plane.EnabledOn()
    
    iren.Initialize()
    iren.Start()
    return
    

def visualizeIsosurface(vtkImage,spacing,isovalue,color,opacity):

    contour = createMesh(vtkImage,isovalue)
    meshActor = createMeshActor(contour,color,opacity)
    outlineActor = createOutlineActor(vtkImage)
    
    # Set Camera Position
    camera = vtk.vtkCamera()
    camera.SetViewUp(0,0,-1)
    camera.SetPosition(-2,-2,-2)
        
    # Create the Renderer, Window and Interator
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.1,0.1,0.2)
    # Actor for outline
    ren.AddActor(outlineActor)
    # Actor for isosurfaces
    ren.AddActor(meshActor)
    # ren.AddActor(meshActor3)
    # Set the camera
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    
    createStartWindow(ren)
    return
    

def visualizeIsosurfacePair(vtkImageF,spacingF,isovalueF,colorF,opacityF,vtkImageM,spacingM,isovalueM,colorM,opacityM):
    contourF = createMesh(vtkImageF,isovalueF)
    meshActorF = createMeshActor(contourF,colorF,opacityF)
    contourM = createMesh(vtkImageM,isovalueM)
    meshActorM = createMeshActor(contourM,colorM,opacityM)

    outlineActor = createOutlineActor(vtkImageF)
    
    # Set Camera Position
    camera = vtk.vtkCamera()
    camera.SetViewUp(0,0,-1)
    camera.SetPosition(-2,-2,-2)
        
    # Create the Renderer, Window and Interator
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.1,0.1,0.2)
    # Actor for outline
    ren.AddActor(outlineActor)
    # Actor for isosurfaces
    ren.AddActor(meshActorF)
    ren.AddActor(meshActorM)
    # Set the camera
    ren.SetActiveCamera(camera)
    ren.ResetCamera()
    
    createStartWindow(ren)
    return



def createFineMesh(vtkData,isovalue=255,trgtRedDecimation=0.75,filterNoOfIterations=20,nMeshPoints=5000):
    """
    Function that takes in vtkImageData and creates a smooth mesh

    Inputs:
    vtkData: Input vtkImageData
    isovalue: Scalar value for isosurface extraction
    trgtReduction: Target reduction of mesh vertices during decimation (0.75 -> reduces the vertices to 75% of original)
    filterNoOfIterations: Number of iterations of vtkWindowedSincPolyDataFilter to apply (20 is a good value)
    
    Outputs:
    meshNormals.GetOutput(): Returns a vtkPolyData mesh
    
    """
    
    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtkData)
    contour.SetValue(0,isovalue)
    contour.Update()
    
    deci = vtk.vtkDecimatePro()
    deci.SetInputConnection(contour.GetOutputPort())
    deci.SetTargetReduction(trgtRedDecimation)
    deci.PreserveTopologyOn()
    # deci.SetFeatureAngle(30)
    deci.SplittingOn()
    deci.Update()
    
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(deci.GetOutputPort())
    smoother.SetNumberOfIterations(filterNoOfIterations)
    # smoother.FeatureEdgeSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.BoundarySmoothingOff()
    smoother.GenerateErrorScalarsOn()
    smoother.Update()
    
    cobj = Clustering.Cluster(smoother.GetOutput())
    cobj.GenClusters(nMeshPoints, subratio=20, verbose=True) 
    # cobj.PlotClusters()

    # Plot new mesh
    cobj.GenMesh()
    # cobj.PlotRemesh()
    reMesh = cobj.ReturnMesh()
    
    # meshNormals = vtk.vtkPolyDataNormals()
    # meshNormals.SetInputData(reMesh)
    # meshNormals.Update()
    # return(meshNormals.GetOutput())
    return(reMesh)


def getPointsPoly(vtkMesh):
    vtkPoints = vtkMesh.GetPoints().GetData()
    numpyPoints = numpy_support.vtk_to_numpy(vtkPoints).astype('double')
    
    vtkCellArray = vtkMesh.GetPolys()
    cells = ()
    
    # meanPoint = np.mean(numpyPoints,axis=0)
    # plt.scatter(numpyPoints[:,0],numpyPoints[:,1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(numpyPoints[:,0],numpyPoints[:,1],numpyPoints[:,2])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    vtkCellArray.InitTraversal()
    idList = vtk.vtkIdList()
    for i in range(vtkCellArray.GetNumberOfCells()):
        vtkCellArray.GetNextCell(idList)
        a = idList.GetId(0)
        b = idList.GetId(1)
        c = idList.GetId(2)
        cells = cells + ((a,b,c),)
    
    return([numpyPoints,cells])


def getSlicesFromMesh(mesh,vtkImage,newSpacing,newSize):
#    newSize = np.floor(np.divide(np.multiply(np.asarray(vtkImage.GetDimensions()),np.asarray(oldSpacing)),np.asarray(newSpacing))).astype(int)
    
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0)
    plane.SetNormal(0, 0, 1)
    
    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)
    cutter.GenerateCutScalarsOn()
    cutter.GenerateValues(newSize[2],0.0,(newSize[2]-1)*newSpacing[2])
    cutter.Update()
    # cutEdges.GenerateTrianglesOn()
    
    cutStrips = vtk.vtkStripper()
    cutStrips.SetInputConnection(cutter.GetOutputPort())
    cutStrips.Update()
    cutPoly = vtk.vtkPolyData()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(cutPoly)
    extrude.SetScaleFactor(newSpacing[2])
    # extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)
    extrude.Update()
    
    imageVTK = vtk.vtkImageData()
    imageVTK.SetSpacing(newSpacing)
    imageVTK.SetDimensions(newSize)
    imageVTK.AllocateScalars(vtk.VTK_TYPE_UINT8,1)
    VTKdata = numpy_support.numpy_to_vtk(255*np.ones(newSize).ravel(), deep=True, array_type=vtk.VTK_TYPE_UINT8)
    imageVTK.GetPointData().SetScalars(VTKdata)
    
    dataToStencil = vtk.vtkPolyDataToImageStencil()
    dataToStencil.SetTolerance(0)
    dataToStencil.SetOutputOrigin(imageVTK.GetOrigin())
    dataToStencil.SetOutputSpacing(newSpacing)
    dataToStencil.SetOutputWholeExtent(imageVTK.GetExtent())
    dataToStencil.SetInputConnection(extrude.GetOutputPort())
    dataToStencil.Update()
    
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(imageVTK)
    stencil.SetStencilConnection(dataToStencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()
    
    cutImage = stencil.GetOutput()
    
    return cutImage


def cutMesh(smoothMesh,newSize,newSpacing):
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 0)
    plane.SetNormal(0, 0, 1)
    
    cutter = vtk.vtkCutter()
    cutter.SetInputData(smoothMesh)
    cutter.SetCutFunction(plane)
    cutter.GenerateCutScalarsOn()
    cutter.GenerateValues(newSize[2],0,(newSize[2]-1)*newSpacing[2])
    cutter.Update()
    # cutEdges.GenerateTrianglesOn()
    
    cutStrips = vtk.vtkStripper()
    cutStrips.SetInputConnection(cutter.GetOutputPort())
    cutStrips.Update()
    cutPoly = vtk.vtkPolyData()
    cutPoly.SetPoints(cutStrips.GetOutput().GetPoints())
    cutPoly.SetPolys(cutStrips.GetOutput().GetLines())

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(cutPoly)
    extrude.SetScaleFactor(newSpacing[2])
    # extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0, 0, 1)
    extrude.Update()
    
    imageVTK = vtk.vtkImageData()
    imageVTK.SetSpacing(newSpacing)
    imageVTK.SetDimensions(newSize)
    imageVTK.SetOrigin(0,0,0)
    imageVTK.AllocateScalars(vtk.VTK_TYPE_UINT8,1)
    VTKdata = numpy_support.numpy_to_vtk(255*np.ones(newSize).ravel(), deep=True, array_type=vtk.VTK_TYPE_UINT8)
    imageVTK.GetPointData().SetScalars(VTKdata)
    
    dataToStencil = vtk.vtkPolyDataToImageStencil()
    dataToStencil.SetTolerance(0)
    dataToStencil.SetOutputOrigin(imageVTK.GetOrigin())
    dataToStencil.SetOutputSpacing(newSpacing)
    dataToStencil.SetOutputWholeExtent(imageVTK.GetExtent())
    dataToStencil.SetInputConnection(extrude.GetOutputPort())
    dataToStencil.Update()
    
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(imageVTK)
    stencil.SetStencilConnection(dataToStencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()
    
    # func.visualizeCompositeVol(stencil.GetOutput())
    
    cutImage = stencil.GetOutput()
    
    vtkImage = cutImage.GetPointData().GetScalars()
    dim = cutImage.GetDimensions()
    newImage = numpy_support.vtk_to_numpy(vtkImage).reshape(dim[-1],dim[-2],dim[-3]).swapaxes(0,2)

    return newImage
    

def interpBasalSlice(GT,oldSpacing,newSpacing):
    distMap = np.zeros(GT.shape)
    for j in range(GT.shape[2]):
        currentSlice = GT[:,:,j]
        indNZ = np.nonzero(currentSlice)
        if(len(indNZ[0])==0):
            continue
        currentSliceEdge = seg.find_boundaries(currentSlice)
        distImg = distance_transform_edt(np.logical_not(currentSliceEdge),return_distances=True)
        distImg[indNZ] = -distImg[indNZ]
        # plt.figure(),plt.imshow(np.logical_not(currentSliceEdge),cmap=plt.cm.gray)
        # plt.figure(),plt.imshow(distImg<0,cmap=plt.cm.gray)
        distMap[:,:,j] = distImg
    
    sitkDist = sitk.GetImageFromArray(distMap.swapaxes(0,2))
    sitkDist.SetSpacing(oldSpacing)
    newSpacing = (oldSpacing[0],oldSpacing[1],1.0)
    # newSize = np.array([GT.shape[0],GT.shape[1],int((GT.shape[2]-1)*oldSpacing[2]/newSpacing[2]+1)])
    rsDist = resampleVolume(sitkDist,newSpacing,'l')
    reVol = sitk.GetArrayFromImage(rsDist).swapaxes(0,2)
    return reVol



def statFeatures(patch):
    feat = scstat.describe(patch.ravel())
    minI,maxI = feat[1]
    mean = feat[2]
    var = feat[3]
    skew = feat[4]
    kurtosis = feat[5]
    entropy = scstat.entropy(patch.ravel(),base=2)
    if(np.isinf(entropy)):
        entropy = 0
    energy = np.sum(patch.ravel()**2)
    return(minI,maxI,mean,var,skew,kurtosis,entropy,energy)
    

def localHOG(patch):
    hogZX = ()
    for i in range(1,patch.shape[0]-1):
        hog = skfeat.hog(patch[i,:,:],orientations=9,pixels_per_cell=patch[i,:,:].shape,cells_per_block=(1,1),visualise=False)
        hogZX = hogZX+tuple(hog)
    hogYZ = ()
    for i in range(1,patch.shape[1]-1):
        hog = skfeat.hog(patch[:,i,:],orientations=9,pixels_per_cell=patch[:,i,:].shape,cells_per_block=(1,1),visualise=False)
        hogYZ = hogYZ+tuple(hog)
    hogXY = ()
    for i in range(1,patch.shape[2]-1):
        hog = skfeat.hog(patch[:,:,i],orientations=9,pixels_per_cell=patch[:,:,i].shape,cells_per_block=(1,1),visualise=False)
        hogXY = hogXY+tuple(hog)
    hogXYZ = hogXY+hogYZ+hogZX
    return(hogXYZ)


def localLBP(patch):
    lbpZX = ()
    for i in range(patch.shape[0]):
        lbpPatch81 = skfeat.local_binary_pattern(patch[i,:,:],8,1,method='uniform')
        # lbpPatch162 = skfeat.local_binary_pattern(patch[i,:,:],16,2,method='uniform')
        # lbpPatch243 = skfeat.local_binary_pattern(patch[i,:,:],24,3,method='uniform')
        lbpZX = lbpZX+(lbpPatch81[1,1]/9,)#lbpPatch162[2,2]/17,lbpPatch243[3,3]/25)
    lbpYZ = ()
    for i in range(patch.shape[1]):
        lbpPatch81 = skfeat.local_binary_pattern(patch[:,i,:],8,1,method='uniform')
        # lbpPatch162 = skfeat.local_binary_pattern(patch[:,i,:],16,2,method='uniform')
        # lbpPatch243 = skfeat.local_binary_pattern(patch[:,i,:],24,3,method='uniform')
        lbpYZ = lbpYZ+(lbpPatch81[1,1]/9,)#lbpPatch162[2,2]/17,lbpPatch243[3,3]/25)
    lbpXY = ()
    for i in range(patch.shape[2]):
        lbpPatch81 = skfeat.local_binary_pattern(patch[:,i,:],8,1,method='uniform')
        # lbpPatch162 = skfeat.local_binary_pattern(patch[:,i,:],16,2,method='uniform')
        # lbpPatch243 = skfeat.local_binary_pattern(patch[:,i,:],24,3,method='uniform')
        lbpXY = lbpXY+(lbpPatch81[1,1]/9,)#lbpPatch162[2,2]/17,lbpPatch243[3,3]/25)
    lbpXYZ = lbpXY+lbpYZ+lbpZX
    return(lbpXYZ)


def spherical2Cart((r,theta,phi)):
    x = int(r*math.sin(math.radians(theta))*math.cos(math.radians(phi)))
    y = int(r*math.sin(math.radians(theta))*math.sin(math.radians(phi)))
    z = int(r*math.cos(math.radians(theta)))
    return(x,y,z)


def randomBox((i,j,k),(p,q,r),(l,b,w)):
    ind = ()
    ind = ind+(i+p-int(np.ceil(float(l)/2)),i+p-int(np.ceil(float(l)/2))+l)
    ind = ind+(j+q-int(np.ceil(float(b)/2)),j+q-int(np.ceil(float(b)/2))+b)
    ind = ind+(k+r-int(np.ceil(float(w)/2)),k+r-int(np.ceil(float(w)/2))+w)
    return(ind)


def boxAvgDiff(integralVol,ijk,rtp1,lbw1,rtp2,lbw2):
    pqr1 = spherical2Cart(rtp1)
#    lbw1 = (3,3,3)
    ind1 = randomBox(ijk,pqr1,lbw1)
    b1 = float(integralVol[ind1[1],ind1[3],ind1[5]]-integralVol[ind1[0],ind1[2],ind1[4]]-
                integralVol[ind1[0],ind1[3],ind1[5]]-integralVol[ind1[1],ind1[2],ind1[5]]-integralVol[ind1[1],ind1[3],ind1[4]]+
                integralVol[ind1[0],ind1[2],ind1[5]]+integralVol[ind1[0],ind1[3],ind1[4]]+integralVol[ind1[1],ind1[2],ind1[4]])/np.prod(lbw1)
    
    pqr2 = spherical2Cart(rtp2)
    ind2 = randomBox(ijk,pqr2,lbw2)
    b2 = float(integralVol[ind2[1],ind2[3],ind2[5]]-integralVol[ind2[0],ind2[2],ind2[4]]-
                integralVol[ind2[0],ind2[3],ind2[5]]-integralVol[ind2[1],ind2[2],ind2[5]]-integralVol[ind2[1],ind2[3],ind2[4]]+
                integralVol[ind2[0],ind2[2],ind2[5]]+integralVol[ind2[0],ind2[3],ind2[4]]+integralVol[ind2[1],ind2[2],ind2[4]])/np.prod(lbw2)
    f = b1-b2
    return(f)


ind1G = ((-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (0, 5, -2, 1, 0, 6),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 3, -4, 3, -2, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (3, 9, -2, 2, -5, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 5, -2, 2, -5, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (1, 7, -2, 5, 2, 5),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (0, 3, -2, 4, -4, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (2, 8, -1, 6, -5, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (2, 6, 0, 3, -8, -1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 1, -3, 4, -3, 4),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 5, 3, 6, -1, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-1, 3, 1, 5, -4, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (0, 6, 1, 8, -9, -3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 2, 2, 8, 3, 7),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 2, 3, 10, -1, 4),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 3, 3, 8, -4, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-4, 3, 1, 4, -5, -1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 0, 1, 8, 2, 9),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 0, 3, 6, -1, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-5, 0, 1, 8, -4, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 1, -3, 4, -5, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 2, -2, 1, -2, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-5, -1, -3, 4, -1, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-9, -4, -1, 6, -6, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 1, -1, 2, -5, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 0, -2, 1, 1, 4),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-12, -5, -2, 2, 0, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-9, -6, -2, 1, -5, 0),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 1, -3, 3, -6, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-5, -2, -3, 1, 1, 5),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-4, 0, -3, 1, -3, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, -2, -4, -1, -4, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, -1, -6, 1, -8, -1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 1, -6, -2, 2, 6),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-6, 0, -9, -4, 0, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-4, -1, -8, -1, -3, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-4, 3, -4, 3, -4, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 2, -3, 0, -1, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 2, -10, -3, -1, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 2, -6, -1, -4, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 3, -7, -4, -8, -3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-1, 5, -7, -2, 1, 7),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 2, -4, 2, -3, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 4, -5, 0, -2, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (1, 5, -9, -2, -9, -4),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 4, -5, 2, 0, 4),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-3, 3, -3, 2, -2, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 2, -4, 3, -2, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 2, -2, 2, -2, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-4, 3, -3, 3, -4, 3),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (2, 5, -2, 2, -1, 2),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (-2, 4, -3, 3, -2, 1),
 (-2, 1, -2, 1, -2, 1),
 (-3, 2, -3, 2, -3, 2),
 (0, 6, -4, 3, -5, -2))
                    
ind2G = ((0, 3, -2, 1, 0, 3),
 (2, 7, -3, 2, 2, 7),
 (0, 3, -3, 3, -1, 5),
 (1, 4, -2, 1, -1, 2),
 (4, 9, -3, 2, -1, 4),
 (1, 6, -2, 2, -2, 4),
 (1, 4, -2, 1, -3, 0),
 (4, 9, -3, 2, -5, 0),
 (1, 5, -2, 1, -4, 1),
 (0, 3, -2, 1, -4, -1),
 (2, 7, -3, 2, -8, -3),
 (-4, 3, -2, 1, -4, 3),
 (0, 3, -1, 2, 0, 3),
 (1, 6, -1, 4, 2, 7),
 (-2, 2, -2, 1, -2, 1),
 (1, 4, -1, 2, -1, 2),
 (3, 8, 0, 5, -1, 4),
 (-2, 5, -1, 3, -4, 3),
 (1, 4, -1, 2, -3, 0),
 (3, 8, 0, 5, -5, 0),
 (2, 7, -1, 4, -3, 0),
 (0, 3, -1, 2, -4, -1),
 (1, 6, -1, 4, -8, -3),
 (-3, 2, -2, 1, -3, 2),
 (-1, 2, 0, 3, 0, 3),
 (-1, 4, 1, 6, 2, 7),
 (-2, 3, -2, 5, -2, 5),
 (-1, 2, 1, 4, -1, 2),
 (0, 5, 3, 8, -1, 4),
 (-1, 6, 4, 7, -2, 5),
 (-1, 2, 1, 4, -3, 0),
 (0, 5, 3, 8, -5, 0),
 (1, 7, 5, 9, -5, 0),
 (-1, 2, 0, 3, -4, -1),
 (-1, 4, 1, 6, -8, -3),
 (0, 3, -1, 6, -6, -2),
 (-2, 1, 0, 3, 0, 3),
 (-3, 2, 2, 7, 2, 7),
 (-4, 3, 1, 5, 0, 5),
 (-2, 1, 1, 4, -1, 2),
 (-3, 2, 4, 9, -1, 4),
 (-3, 3, 1, 7, -1, 2),
 (-2, 1, 1, 4, -3, 0),
 (-3, 2, 4, 9, -5, 0),
 (-2, 2, 1, 7, -5, 2),
 (-2, 1, 0, 3, -4, -1),
 (-3, 2, 2, 7, -8, -3),
 (-2, 1, 2, 7, -9, -2),
 (-3, 0, 0, 3, 0, 3),
 (-5, 0, 1, 6, 2, 7),
 (-3, 3, -2, 1, -3, 3),
 (-3, 0, 1, 4, -1, 2),
 (-6, -1, 3, 8, -1, 4),
 (-4, 3, -2, 1, -3, 3),
 (-3, 0, 1, 4, -3, 0),
 (-6, -1, 3, 8, -5, 0),
 (-6, -2, 4, 9, -5, 0),
 (-3, 0, 0, 3, -4, -1),
 (-5, 0, 1, 6, -8, -3),
 (-5, 2, 0, 5, -6, 0),
 (-4, -1, -1, 2, 0, 3),
 (-7, -2, -1, 4, 2, 7),
 (-3, 2, -2, 2, -3, 2),
 (-5, -2, -1, 2, -1, 2),
 (-9, -4, 0, 5, -1, 4),
 (-3, 0, -2, 1, -3, 3),
 (-5, -2, -1, 2, -3, 0),
 (-9, -4, 0, 5, -5, 0),
 (-3, 3, -3, 3, -2, 2),
 (-4, -1, -1, 2, -4, -1),
 (-7, -2, -1, 4, -8, -3),
 (-6, 0, -1, 2, -5, -2),
 (-4, -1, -2, 1, 0, 3),
 (-8, -3, -3, 2, 2, 7),
 (-8, -3, -3, 2, 2, 7),
 (-5, -2, -2, 1, -1, 2),
 (-10, -5, -3, 2, -1, 4),
 (-9, -5, -4, 3, -2, 5),
 (-5, -2, -2, 1, -3, 0),
 (-10, -5, -3, 2, -5, 0),
 (-3, 0, -2, 1, -3, 2),
 (-4, -1, -2, 1, -4, -1),
 (-8, -3, -3, 2, -8, -3),
 (-4, -1, -2, 1, -4, 0),
 (-4, -1, -3, 0, 0, 3),
 (-7, -2, -5, 0, 2, 7),
 (-7, -1, -5, 1, 3, 7),
 (-5, -2, -3, 0, -1, 2),
 (-9, -4, -6, -1, -1, 4),
 (-11, -4, -7, -1, -1, 4),
 (-5, -2, -3, 0, -3, 0),
 (-9, -4, -6, -1, -5, 0),
 (-9, -2, -6, -1, -3, 0),
 (-4, -1, -3, 0, -4, -1),
 (-7, -2, -5, 0, -8, -3),
 (-4, 3, -4, 3, -2, 1),
 (-3, 0, -4, -1, 0, 3),
 (-5, 0, -7, -2, 2, 7),
 (-3, 2, -2, 1, -2, 2),
 (-3, 0, -5, -2, -1, 2),
 (-6, -1, -9, -4, -1, 4),
 (-5, -2, -8, -3, -2, 3),
 (-3, 0, -5, -2, -3, 0),
 (-6, -1, -9, -4, -5, 0),
 (-6, -2, -11, -4, -6, 1),
 (-3, 0, -4, -1, -4, -1),
 (-5, 0, -7, -2, -8, -3),
 (-2, 1, -2, 2, -3, 2),
 (-2, 1, -4, -1, 0, 3),
 (-3, 2, -8, -3, 2, 7),
 (-4, 3, -5, -2, -1, 6),
 (-2, 1, -5, -2, -1, 2),
 (-3, 2, -10, -5, -1, 4),
 (-2, 2, -8, -4, -1, 3),
 (-2, 1, -5, -2, -3, 0),
 (-3, 2, -10, -5, -5, 0),
 (-3, 2, -12, -5, -5, 1),
 (-2, 1, -4, -1, -4, -1),
 (-3, 2, -8, -3, -8, -3),
 (-2, 1, -6, 0, -6, 0),
 (-1, 2, -4, -1, 0, 3),
 (-1, 4, -7, -2, 2, 7),
 (-2, 1, -2, 2, -2, 2),
 (-1, 2, -5, -2, -1, 2),
 (0, 5, -9, -4, -1, 4),
 (-2, 2, -2, 1, -3, 2),
 (-1, 2, -5, -2, -3, 0),
 (0, 5, -9, -4, -5, 0),
 (-1, 2, -5, 1, -2, 1),
 (-1, 2, -4, -1, -4, -1),
 (-1, 4, -7, -2, -8, -3),
 (-3, 2, -2, 1, -3, 3),
 (0, 3, -3, 0, 0, 3),
 (1, 6, -5, 0, 2, 7),
 (2, 6, -6, 1, 2, 5),
 (1, 4, -3, 0, -1, 2),
 (3, 8, -6, -1, -1, 4),
 (0, 6, -4, 1, -3, 4),
 (1, 4, -3, 0, -3, 0),
 (3, 8, -6, -1, -5, 0),
 (1, 5, -3, 1, -3, 1),
 (0, 3, -3, 0, -4, -1),
 (1, 6, -5, 0, -8, -3),
 (-3, 3, -3, 2, -2, 2),
 (0, 3, -2, 1, 0, 3),
 (2, 7, -3, 2, 2, 7),
 (-3, 3, -2, 1, -2, 1),
 (1, 4, -2, 1, -1, 2),
 (4, 9, -3, 2, -1, 4),
 (-2, 3, -2, 2, -2, 1),
 (1, 4, -2, 1, -3, 0),
 (4, 9, -3, 2, -5, 0),
 (-3, 3, -4, 3, -4, 3),
 (0, 3, -2, 1, -4, -1),
 (2, 7, -3, 2, -8, -3),
 (-2, 2, -2, 1, -3, 3))


def contextualFeatures(integralVol,i,j,k):
    global ind1G
    b1 = ()
    for ind1 in ind1G:
        ind1 = (ind1[0]+i,ind1[1]+i,ind1[2]+j,ind1[3]+j,ind1[4]+k,ind1[5]+k)
        b1 = b1+(float(integralVol[ind1[1],ind1[3],ind1[5]]-integralVol[ind1[0],ind1[2],ind1[4]]-
                    integralVol[ind1[0],ind1[3],ind1[5]]-integralVol[ind1[1],ind1[2],ind1[5]]-integralVol[ind1[1],ind1[3],ind1[4]]+
                    integralVol[ind1[0],ind1[2],ind1[5]]+integralVol[ind1[0],ind1[3],ind1[4]]+integralVol[ind1[1],ind1[2],ind1[4]])
                    /((ind1[1]-ind1[0])*(ind1[3]-ind1[2])*(ind1[5]-ind1[4])),)
    global ind2G
    b2 = ()
    for ind2 in ind2G:
        ind2 = (ind2[0]+i,ind2[1]+i,ind2[2]+j,ind2[3]+j,ind2[4]+k,ind2[5]+k)
        b2 = b2+(float(integralVol[ind2[1],ind2[3],ind2[5]]-integralVol[ind2[0],ind2[2],ind2[4]]-
                    integralVol[ind2[0],ind2[3],ind2[5]]-integralVol[ind2[1],ind2[2],ind2[5]]-integralVol[ind2[1],ind2[3],ind2[4]]+
                    integralVol[ind2[0],ind2[2],ind2[5]]+integralVol[ind2[0],ind2[3],ind2[4]]+integralVol[ind2[1],ind2[2],ind2[4]])
                    /((ind2[1]-ind2[0])*(ind2[3]-ind2[2])*(ind2[5]-ind2[4])),)

    f = np.asarray(b1)-np.asarray(b2)
    return(tuple(f))

## Function to import a python array as vtk image
#def vtkDataImport(data_matrix,spacing):
#    # Inputs:
#    # data_matrix: N-Dimensional python array (eg 3D image)
#    # spacing: Physical pixel spacing for the input image (zSpacing,ySpacing,xSpacing)
#
#    # Outputs:
#    # dataImporter: VTK-image with set attributes
#
#
#    # For VTK to be able to use the data, it must be stored as a VTK-image. 
#    # This can be done by the vtkImageImport-class which imports raw data and 
#    # stores it.
#    dataImporter = vtk.vtkImageImport()
#
#    # The previously created array is converted to a string of chars and imported.
#    data_string = data_matrix.tostring()
#    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
#    # The type of the newly imported data is set to unsigned char (uint8)
#    dataImporter.SetDataScalarTypeToUnsignedChar()
#    # Because the data that is imported only contains an intensity value (it isnt 
#    # RGB-coded or someting similar), the importer must be told this is the case.
#    dataImporter.SetNumberOfScalarComponents(1)
#    dataImporter.SetDataSpacing(spacing[0],spacing[1],spacing[2])
#    # The following two functions describe how the data is stored and the 
#    # dimensions of the array it is stored in. For this simple case, all axes are 
#    # of length 75 and begins with the first element. For other data, this is 
#    # probably not the case. I have to admit however, that I honestly dont know 
#    # the difference between SetDataExtent() and SetWholeExtent() although
#    # VTK complains if not both are used.
#    w, h, d = data_matrix.shape
#    dataImporter.SetDataExtent(0, d-1, 0, h-1, 0, w-1)
#    dataImporter.SetWholeExtent(0, d-1, 0, h-1, 0, w-1)
#    return dataImporter


    
## Function to visualize the Iso-surface of a given volume    
#def visualizeIsosurface(data_matrix,spacing,isovalue):
##    if(len(data_matrix.shape)<=3):
##        dataImporter = importArray(data_matrix,1,spacing)
##    else:
##        dataImporter = importArray(data_matrix,3,spacing)
#    dataImporter = importArray(data_matrix,1,spacing)
#        
#    alphaChannelFunc = vtk.vtkPiecewiseFunction()
#    colorFunc = vtk.vtkColorTransferFunction()
#    maxI = data_matrix.max()
#    for i in range(int(maxI)+1):
#        alphaChannelFunc.AddPoint(i, 0.1)
#        intensity = i/float(maxI)
#        colorFunc.AddRGBPoint(i,intensity,intensity,intensity)
#    
#    # Set black opacity to 0 to see the sample
#    alphaChannelFunc.AddPoint(0,0.0)
#    
#    # The preavius two classes stored properties. Because we want to apply these
#    # properties to the volume we want to render, we have to store them in a class 
#    # that stores volume prpoperties.
#    volumeProperty = vtk.vtkVolumeProperty()
#
##    if(dataImporter.GetDataDimension()<=3):
##        volumeProperty.SetColor(colorFunc)
##    else:
##        volumeProperty.SetColor(0,colorFunc)
##        volumeProperty.SetColor(1,colorFunc)
##        volumeProperty.SetColor(2,colorFunc)
#
#    volumeProperty.SetColor(colorFunc)
#    volumeProperty.SetScalarOpacity(alphaChannelFunc)
#    volumeProperty.ShadeOn()
#    
#    # This class describes how the volume is rendered (through ray tracing).
#    
#    ## For Isosurface
#    isoFunction = vtkVolumeRayCastIsosurfaceFunction()
#    isoFunction.SetIsoValue(isovalue)
#    
#    # function to reduce the spacing between each image
#    #volumeMapper.SetMaximumImageSampleDistance(0.01)
#    
#    # We can finally create our volume. We also have to specify the data for it, 
#    # as well as how the data will be rendered.
#    volumeMapper = vtk.vtkVolumeRayCastMapper()
#    volumeMapper.SetVolumeRayCastFunction(isoFunction)
#    volumeMapper.SetInputData(dataImporter)
#    
#    # The class vtkVolume is used to pair the previously declared volume as well 
#    # as the properties to be used when rendering that volume.
#    volume = vtk.vtkVolume()
#    volume.SetMapper(volumeMapper)
#    volume.SetProperty(volumeProperty)
#    
#    # Create an outline
#    outlineData = vtkOutlineFilter()
#    outlineData.SetInputData(dataImporter)
#    outlineMapper = vtkPolyDataMapper()
#    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
#    outline = vtkActor()
#    outline.SetMapper(outlineMapper)
#    outline.GetProperty().SetColor(0.9,0.9,0.9)
#    
#    # Set Camera Position
#    camera = vtkCamera()
#    camera.SetViewUp(0,0,-1)
#    camera.SetPosition(-2,-2,-2)
#        
#    ## Initialize a Plane
#    #plane = vtkImagePlaneWidget()
#    #plane.SetInputConnection(dataImporter.GetOutputPort())
#    #plane.SetSliceIndex(3)
#    
#    # Create the Renderer, Window and Interator
#    ren = vtk.vtkRenderer()
#    ren.AddVolume(volume)
#    ren.SetBackground(0.1,0.1,0.2)
#    # Actor for outline
#    ren.AddActor(outline)
#    # Set the camera
#    ren.SetActiveCamera(camera)
#    ren.ResetCamera()
#    
#    # Create the window
#    renWin = vtk.vtkRenderWindow()
#    renWin.AddRenderer(ren)
#    renWin.SetWindowName("Volume rendering")
#    renWin.SetSize(1000,1000)
#    
#    # Start the application
#    iren = vtkRenderWindowInteractor()
#    iren.SetRenderWindow(renWin)
#
##    # For cut plane
##    plane.SetPlaneOrientationToXAxes()
##    #plane.SetPlaneOrientationToYAxes()
##    #plane.SetPlaneOrientationToZAxes()
##    volume.VisibilityOff()
##    plane.SetInteractor(iren)
##    plane.EnabledOn()
#    
#    iren.Initialize()
#    iren.Start()
#    return


## Function to visualize the Iso-surface of a given volume    
#def visualizeIsosurface(data_matrix,spacing,isovalue):
#    dataImporter = importArray(data_matrix.swapaxes(0,2),1,spacing)
#        
#    # Do surface rendering
#    isosurfaceExtractor = vtk.vtkMarchingCubes()
#    isosurfaceExtractor.SetInputData(dataImporter)
#    isosurfaceExtractor.SetValue(0,240)
#    
#    isosurfaceNormals = vtk.vtkPolyDataNormals()
#    isosurfaceNormals.SetInputConnection(isosurfaceExtractor.GetOutputPort())
#    isosurfaceNormals.SetFeatureAngle(60.0)
#    
#    isosurfaceStripper = vtk.vtkStripper()
#    isosurfaceStripper.SetInputConnection(isosurfaceNormals.GetOutputPort())
#    
#    isosurfaceMapper = vtk.vtkPolyDataMapper()
#    isosurfaceMapper.SetInputConnection(isosurfaceStripper.GetOutputPort())
#    isosurfaceMapper.ScalarVisibilityOff()
#    
#    isosurfaceProperty = vtk.vtkProperty()
#    isosurfaceProperty.SetColor(1.0,0,0)
#    isosurfaceProperty.SetOpacity(0.5)
#    
#    isosurface = vtk.vtkActor()
#    isosurface.SetMapper(isosurfaceMapper)
#    isosurface.SetProperty(isosurfaceProperty)
#
#    # Create an outline
#    outlineData = vtkOutlineFilter()
#    outlineData.SetInputData(dataImporter)
#    outlineMapper = vtkPolyDataMapper()
#    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
#    outline = vtkActor()
#    outline.SetMapper(outlineMapper)
#    outline.GetProperty().SetColor(0.9,0.9,0.9)
#    
#    # Set Camera Position
#    camera = vtkCamera()
#    camera.SetViewUp(0,0,-1)
#    camera.SetPosition(-2,-2,-2)
#        
#    
#    # Create the Renderer, Window and Interator
#    ren = vtk.vtkRenderer()
#    ren.SetBackground(0.1,0.1,0.2)
#    # Actor for outline
#    ren.AddActor(outline)
#    # Actor for isosurface
#    ren.AddActor(isosurface)
#    # Set the camera
#    ren.SetActiveCamera(camera)
#    ren.ResetCamera()
#    
#    # Create the window
#    renWin = vtk.vtkRenderWindow()
#    renWin.AddRenderer(ren)
#    renWin.SetWindowName("Volume rendering")
#    renWin.SetSize(1000,1000)
#    
#    # Start the application
#    iren = vtkRenderWindowInteractor()
#    iren.SetRenderWindow(renWin)
#
##    # For cut plane
##    plane.SetPlaneOrientationToXAxes()
##    #plane.SetPlaneOrientationToYAxes()
##    #plane.SetPlaneOrientationToZAxes()
##    volume.VisibilityOff()
##    plane.SetInteractor(iren)
##    plane.EnabledOn()
#    
#    iren.Initialize()
#    iren.Start()
#    return


## Function to visualize the Iso-surface of a given volume    
#def visualizeIsosurfacePair(data_matrix_fixed,data_matrix_moving,spacing,isovalue):
#
#    dataImporterF = importArray(data_matrix_fixed.swapaxes(0,2),1,spacing)
#    dataImporterM = importArray(data_matrix_moving.swapaxes(0,2),1,spacing)    
#
#    # Do surface rendering
#    isosurfaceExtractorF = vtk.vtkMarchingCubes()
#    isosurfaceExtractorF.SetInputData(dataImporterF)
#    isosurfaceExtractorF.SetValue(0,isovalue)
#    
#    isosurfaceNormalsF = vtk.vtkPolyDataNormals()
#    isosurfaceNormalsF.SetInputConnection(isosurfaceExtractorF.GetOutputPort())
#    isosurfaceNormalsF.SetFeatureAngle(60.0)
#    
#    isosurfaceStripperF = vtk.vtkStripper()
#    isosurfaceStripperF.SetInputConnection(isosurfaceNormalsF.GetOutputPort())
#    
#    isosurfaceMapperF = vtk.vtkPolyDataMapper()
#    isosurfaceMapperF.SetInputConnection(isosurfaceStripperF.GetOutputPort())
#    isosurfaceMapperF.ScalarVisibilityOff()
#    
#    isosurfacePropertyF = vtk.vtkProperty()
#    isosurfacePropertyF.SetColor(1.0,0,0)
#    isosurfacePropertyF.SetOpacity(0.3)
#    
#    isosurfaceF = vtk.vtkActor()
#    isosurfaceF.SetMapper(isosurfaceMapperF)
#    isosurfaceF.SetProperty(isosurfacePropertyF)
#    
#    
#    # Do surface rendering
#    isosurfaceExtractorM = vtk.vtkMarchingCubes()
#    isosurfaceExtractorM.SetInputData(dataImporterM)
#    isosurfaceExtractorM.SetValue(0,isovalue)
#    
#    isosurfaceNormalsM = vtk.vtkPolyDataNormals()
#    isosurfaceNormalsM.SetInputConnection(isosurfaceExtractorM.GetOutputPort())
#    isosurfaceNormalsM.SetFeatureAngle(60.0)
#    
#    isosurfaceStripperM = vtk.vtkStripper()
#    isosurfaceStripperM.SetInputConnection(isosurfaceNormalsM.GetOutputPort())
#    
#    isosurfaceMapperM = vtk.vtkPolyDataMapper()
#    isosurfaceMapperM.SetInputConnection(isosurfaceStripperM.GetOutputPort())
#    isosurfaceMapperM.ScalarVisibilityOff()
#    
#    isosurfacePropertyM = vtk.vtkProperty()
#    isosurfacePropertyM.SetColor(0,0,1.0)
#    isosurfacePropertyM.SetOpacity(0.4)
#    
#    isosurfaceM = vtk.vtkActor()
#    isosurfaceM.SetMapper(isosurfaceMapperM)
#    isosurfaceM.SetProperty(isosurfacePropertyM)
#
#    
#    # Create an outline
#    outlineData = vtkOutlineFilter()
#    outlineData.SetInputData(dataImporterF)
#    outlineMapper = vtkPolyDataMapper()
#    outlineMapper.SetInputConnection(outlineData.GetOutputPort())
#    outline = vtkActor()
#    outline.SetMapper(outlineMapper)
#    outline.GetProperty().SetColor(0.9,0.9,0.9)
#    
#    # Set Camera Position
#    camera = vtkCamera()
#    camera.SetViewUp(0,0,-1)
#    camera.SetPosition(-2,-2,-2)
#        
#    ## Initialize a Plane
#    #plane = vtkImagePlaneWidget()
#    #plane.SetInputConnection(dataImporter.GetOutputPort())
#    #plane.SetSliceIndex(3)
#    
#    # Create the Renderer, Window and Interator
#    ren = vtk.vtkRenderer()
#    ren.SetBackground(0.1,0.1,0.2)
#    # Actor for outline
#    ren.AddActor(outline)
#    # Actor for isosurfaces
#    ren.AddActor(isosurfaceF)
#    ren.AddActor(isosurfaceM)
#    # Set the camera
#    ren.SetActiveCamera(camera)
#    ren.ResetCamera()
#    
#    # Create the window
#    renWin = vtk.vtkRenderWindow()
#    renWin.AddRenderer(ren)
#    renWin.SetWindowName("Volume rendering")
#    renWin.SetSize(1000,1000)
#    
#    # Start the application
#    iren = vtkRenderWindowInteractor()
#    iren.SetRenderWindow(renWin)
#
##    # For cut plane
##    plane.SetPlaneOrientationToXAxes()
##    #plane.SetPlaneOrientationToYAxes()
##    #plane.SetPlaneOrientationToZAxes()
##    volume.VisibilityOff()
##    plane.SetInteractor(iren)
##    plane.EnabledOn()
##    volumeM.GetProperty().SetOpacity(0.5)    
#    
#    iren.Initialize()
#    iren.Start()
#    return
