# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:00:43 2016

@author: sxd7257
"""
import sys
sys.path.append("C:\Users\sxd7257\Dropbox\Python Scripts")
import numpy as np
#import matplotlib.pyplot as plt
import myFunctions as func
import pickle
import scipy.io as sio
import SimpleITK as sitk
import matplotlib.pyplot as plt
#import myFunctions as func
from tables import *
import skimage.util as skutil
#import time
from vtk import *
import scipy.ndimage.morphology as morph
import math


#matContents = sio.loadmat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\avgVolGTIntensity.mat')
##matContents = sio.loadmat('avgVolGT.mat')
#keys = matContents.keys()
#for i in keys:
#    exec(i+"= matContents['"+i+"']")

#matContents = sio.loadmat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\resampledVols.mat')
#keys = matContents.keys()
#for i in keys:
#    exec(i+"= matContents['"+i+"']")

#IOP = sio.loadmat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\ImageOrientations.mat')

#exec("fileName = 'LVMasks.mat'")
#matContents = sio.loadmat('N:\\ShusilDangi\\MICCAI_Segmentation_Challenge\\MatFiles\\'+fileName)
#keys = matContents.keys()
#for i in keys:
#    exec(i+"= matContents['"+i+"']")


def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))


def findOptimumTform(moving_image,fixed_image,movingMask=[],fixedMask=[],verbose=True):
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,moving_image,sitk.Euler3DTransform(),sitk.CenteredTransformInitializerFilter.MOMENTS)
    optimized_transform = sitk.AffineTransform(3)

    registration_method = sitk.ImageRegistrationMethod()
    
    
    if(len(movingMask)>0):
        registration_method.SetMetricMovingMask(movingMask)

    if(len(fixedMask)>0):
        registration_method.SetMetricFixedMask(fixedMask)

    #similarity metric settings
#    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=20)
    registration_method.SetMetricAsMeanSquares()
#    registration_method.SetMetricAsCorrelation()
#    registration_method.SetMetricAsANTSNeighborhoodCorrelation(5)
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)
#    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
#    registration_method.SetMetricSamplingPercentage(0.3)
    
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    #optimizer settings
#    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10, estimateLearningRate=registration_method.EachIteration, maximumStepSizeInPhysicalUnits=1.0)
#    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,minStep=1e-4,numberOfIterations=1000,relaxationFactor=0.5,gradientMagnitudeTolerance=1e-4,estimateLearningRate=registration_method.EachIteration,maximumStepSizeInPhysicalUnits=0.0)
    registration_method.SetOptimizerAsAmoeba(simplexDelta=1.0, numberOfIterations=1000, parametersConvergenceTolerance = 1e-8, functionConvergenceTolerance = 1e-4, withRestarts = True)
#    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=0.05,maximumNumberOfCorrections=5,maximumNumberOfFunctionEvaluations=2000,costFunctionConvergenceFactor=1e+7,lowerBound=0.0,upperBound=0.0,trace=True)
    registration_method.SetOptimizerScalesFromPhysicalShift()
#    registration_method.SetOptimizerScales((1.0,1.0,0.1,0.1))    
    
    #setup for the multi-resolution framework            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2,0])
#    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    #don't optimize in-place, we would possibly like to run this cell multiple times
    #registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)
#    registration_method.SetOptimizerWeights((1.0,0.0,1.0,1.0))
    registration_method.SetOptimizerWeights((1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.1))
    
    #connect all of the observers so that we can perform plotting during registration
    #registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    #registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    #registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    if(verbose):
        registration_method.AddCommand(sitk.sitkIterationEvent,lambda: command_iteration(registration_method))    
    
    #final_transform = registration_method.Execute(fixed_image,moving_image)
    #registration_method.Execute(fixed_GT,moving_GT)
    registration_method.Execute(fixed_image,moving_image)
    final_transform = sitk.Transform(optimized_transform)
    final_transform.AddTransform(initial_transform)
#    tx = final_transform.GetParameters()
#    tx = optimized_transform.GetTranslation()
#    tx = optimized_transform.GetOffset()
#    tform = sitk.Transform(optimized_transform)
    # Convert Translation parameters into integer to avoid interpolation
#    txInt = ()
#    for i in tx:
#        txInt = txInt+(round(i),)
#    final_transform.SetParameters(txInt)
#    final_transform.SetOffset(txInt)
    
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#    print('Translation Parameters, {0}'.format(tx))
#    print('Scale, Rotation, Translation, {0}'.format(tform.GetParameters()))
    ##########################################################
        
    return (final_transform,optimized_transform,initial_transform)    


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
#    axisFixed = np.cross(IOPFixed[0:3],IOPFixed[3:6])
    angle = angle_between_vectors(IOPFixed[0:3],IOPMoving[0:3],directed=False)
    rotation1 = sitk.VersorTransform(tuple(axisMoving),angle)
    rotation2 = sitk.VersorTransform(tuple(axisMoving),-angle)
    tPoint1 = rotation1.TransformPoint(IOPFixed[0:3])
    tPoint2 = rotation2.TransformPoint(IOPFixed[0:3])
    if(np.sum(np.abs(np.asarray(tPoint1)-IOPMoving[0:3]))<np.sum(np.abs(np.asarray(tPoint2)-IOPMoving[0:3]))):
        rotation = sitk.VersorTransform((0,0,1),-angle)
    else:
        rotation = sitk.VersorTransform((0,0,1),angle)
    return rotation
    
    

ind = 12
exec("fileName = 'rawData"+str(ind)+".mat'")
matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\'+fileName)
keys = matContents.keys()
for i in keys:
    exec(i+"= matContents['"+i+"']")
v = np.copy(func.stretchContrast(vol[:,:,:,0],0,99))
spacingFixed = np.copy(spacing[0])
sitkFixedVol = sitk.GetImageFromArray(v.swapaxes(0,2))
sitkFixedVol = sitk.Cast(sitkFixedVol,sitk.sitkFloat32)
sitkFixedVol.SetSpacing(spacingFixed)
sitkFixedGT = sitk.GetImageFromArray(255*bp[:,:,:,0].swapaxes(0,2))
sitkFixedGT = sitk.Cast(sitkFixedGT,sitk.sitkFloat32)
sitkFixedGT.SetSpacing(spacingFixed)
#exec('mask = LVMask'+str(ind))
#maskVol = func.getCircularMask(mask,vol.shape[0:3])
indNZ = np.nonzero(bp[:,:,:,0])
startInd1 = (0.9*np.min(indNZ,axis=1)).astype(int)
endInd1 = (1.1*np.max(indNZ,axis=1)).astype(int)+1
endInd1[2] = min(endInd1[2],bp.shape[2])#startInd1 = (0.9*np.min(indNZ,axis=1)).astype(int)
#endInd1 = (1.1*np.max(indNZ,axis=1)).astype(int)+1

#cropVol = func.normalizeSlices(v[startInd1[0]:endInd1[0],int(0.8*startInd1[1]):int(1.05*endInd1[1]),startInd1[2]:endInd1[2]])
cropVol = func.normalizeSlices(v[startInd1[0]:endInd1[0],startInd1[1]:endInd1[1],startInd1[2]:endInd1[2]])
#cropVol[:,:,0:startInd1[2]] = 0
#bSlice = np.zeros((cropVol.shape[0],cropVol.shape[1]))
#cropVol = np.dstack((bSlice,bSlice,cropVol,bSlice,bSlice))
#func.displayMontage(cropVol,5)
#cropVol[:,:,endInd1[2]:] = 0
#cropVol = vol[startInd1[0]:endInd1[0],startInd1[1]:endInd1[1],startInd1[2]:endInd1[2],0]
fixedImg = np.copy(cropVol)
#fixedGT = gt[startInd1[0]:endInd1[0],int(0.8*startInd1[1]):int(1.05*endInd1[1]),startInd1[2]:endInd1[2],0]
fixedGT = 255*bp[startInd1[0]:endInd1[0],startInd1[1]:endInd1[1],startInd1[2]:endInd1[2],0]
#fixedGT = np.dstack((bSlice,bSlice,fixedGT,bSlice,bSlice))
#func.displayMontage(255*fixedGT,5)
#fixedMask = maskVol[startInd1[0]:endInd1[0],startInd1[1]:endInd1[1],:]
#fixedGTInv = np.logical_not(fixedGT>0.5)
#distFixedGT = morph.distance_transform_edt(fixedGTInv,sampling=spacing[0],return_distances=True,return_indices=False)

fixed_image = sitk.GetImageFromArray(fixedImg.swapaxes(0,2))
fixed_image = sitk.Cast(fixed_image,sitk.sitkFloat32)
fixed_image.SetSpacing(spacingFixed)
fixed_GT = sitk.GetImageFromArray(fixedGT.swapaxes(0,2))
fixed_GT = sitk.Cast(fixed_GT,sitk.sitkFloat32)
fixed_GT.SetSpacing(spacingFixed)
IOPFixed = IOP[0]

#c = 5.0
#r = np.ceil(float(fixedImg.shape[2])/c)
#f,axarr = plt.subplots(int(r),int(c))
#for i in range(fixedImg.shape[2]):
#    x,y = np.unravel_index(i,(int(r),int(c)))
#    axarr[x,y].hist(fixedImg[:,:,i].ravel())
#plt.show()

#fixedGT_dist = sitk.GetImageFromArray(distFixedGT.swapaxes(0,2))
#fixedGT_dist = sitk.Cast(fixedGT_dist,sitk.sitkFloat32)
#fixedGT_dist.SetSpacing(spacing[0])
#fixed_mask = sitk.GetImageFromArray(fixedMask.swapaxes(0,2))
#fixed_mask = sitk.Cast(fixed_mask,sitk.sitkUInt8)
#fixed_mask.SetSpacing(spacing[0])
#func.displayMontage(fixedImg,5)
#func.displayMontage(distFixedGT,5)
#func.displayMontage(sitk.GetArrayFromImage(fixed_image).swapaxes(0,2),5)
#func.displayMontage(sitk.GetArrayFromImage(sitkFixedVol).swapaxes(0,2),5)

## Histogram Matching Filter
histFilter = sitk.HistogramMatchingImageFilter()
histFilter.SetNumberOfHistogramLevels(256)
histFilter.SetNumberOfMatchPoints(5)
histFilter.SetThresholdAtMeanIntensity(True)

dictVar = {}

#indNG = [40,42,99]   # Patients with variable array sizes
indices = range(0,16)
#for i in indNG: indices.remove(i)

for ind in indices:
    # Moving Volume
#    ind = 4
    exec("fileName = 'rawData"+str(ind)+".mat'")
    matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\'+fileName)
    keys = matContents.keys()
    for i in keys:
        exec(i+"= matContents['"+i+"']")
    v = np.copy(func.stretchContrast(vol[:,:,:,0],0,99))
    spacingMoving = np.copy(spacing[0])
    sitkMovingVol = sitk.GetImageFromArray(v.swapaxes(0,2))
    sitkMovingVol = sitk.Cast(sitkMovingVol,sitk.sitkFloat32)
    sitkMovingVol.SetSpacing(spacingMoving)
    sitkMovingGT = sitk.GetImageFromArray(255*bp[:,:,:,0].swapaxes(0,2))
    sitkMovingGT = sitk.Cast(sitkMovingGT,sitk.sitkFloat32)
    sitkMovingGT.SetSpacing(spacingMoving)
    IOPMoving = IOP[0]
#    func.displayMontage(sitk.GetArrayFromImage(sitkMovingVol).swapaxes(0,2),5)
    ## Correct the orientation before registration
    rotation = findRotation(IOPMoving,IOPFixed)
    centerMoving = tuple(map(lambda x,y:x*y/2,sitkMovingVol.GetSize(),sitkMovingVol.GetSpacing()))
    rotation.SetCenter(centerMoving)
    sitkMovingVol = sitk.Resample(sitkMovingVol, sitkMovingVol, rotation, sitk.sitkLinear, 0.0, sitkMovingVol.GetPixelIDValue())
    sitkMovingGT = sitk.Resample(sitkMovingGT, sitkMovingGT, rotation, sitk.sitkLinear, 0.0, sitkMovingGT.GetPixelIDValue())
#    func.displayMontage(sitk.GetArrayFromImage(sitkMovingVol).swapaxes(0,2),5)
#    exec('mask = LVMask'+str(ind))
#    maskVol = func.getCircularMask(mask,vol.shape[0:3],scale=1.15)
    rotVol = sitk.GetArrayFromImage(sitkMovingVol).swapaxes(0,2)
    rotGT = sitk.GetArrayFromImage(sitkMovingGT).swapaxes(0,2)
    indNZ = np.nonzero(rotGT)
    startInd2 = (0.9*np.min(indNZ,axis=1)).astype(int)
    endInd2 = (1.1*np.max(indNZ,axis=1)).astype(int)+1
    endInd2[2] = min(endInd2[2],bp.shape[2])
#    cropVol = func.normalizeSlices(rotVol[startInd2[0]:endInd2[0],int(0.8*startInd2[1]):int(1.05*endInd2[1]),startInd2[2]:endInd2[2]])
    cropVol = func.normalizeSlices(rotVol[startInd2[0]:endInd2[0],startInd2[1]:endInd2[1],startInd2[2]:endInd2[2]])
#    bSlice = np.zeros((cropVol.shape[0],cropVol.shape[1]))
#    cropVol = np.dstack((bSlice,bSlice,cropVol,bSlice,bSlice))
#    func.displayMontage(cropVol,5)
#    func.displayMontage(v,5)
#    func.displayMontage(rotVol,5)
    
    movingImg = np.copy(cropVol)
#    movingGT = rotGT[startInd2[0]:endInd2[0],int(0.8*startInd2[1]):int(1.05*endInd2[1]),startInd2[2]:endInd2[2]]
    movingGT = rotGT[startInd2[0]:endInd2[0],startInd2[1]:endInd2[1],startInd2[2]:endInd2[2]]
#    movingGT = np.dstack((bSlice,bSlice,movingGT,bSlice,bSlice))
#    func.displayMontage(movingGT,5)
#    movingMask = maskVol[startInd2[0]:endInd2[0],startInd2[1]:endInd2[1],:]
#    movingGTInv = np.logical_not(movingGT>0.5)
#    distMovingGT = morph.distance_transform_edt(movingGTInv,sampling=spacing[0],return_distances=True,return_indices=False)
    moving_image = sitk.GetImageFromArray(movingImg.swapaxes(0,2))
    moving_image = sitk.Cast(moving_image,sitk.sitkFloat32)
    moving_image.SetSpacing(spacingMoving)
    moving_image = histFilter.Execute(moving_image,fixed_image)
    movingImg = sitk.GetArrayFromImage(moving_image).swapaxes(0,2)
    moving_GT = sitk.GetImageFromArray(movingGT.swapaxes(0,2))
    moving_GT = sitk.Cast(moving_GT,sitk.sitkFloat32)
    moving_GT.SetSpacing(spacingMoving)

#    movingGT_dist = sitk.GetImageFromArray(distMovingGT.swapaxes(0,2))
#    movingGT_dist = sitk.Cast(movingGT_dist,sitk.sitkFloat32)
#    movingGT_dist.SetSpacing(spacing[0])
#    moving_mask = sitk.GetImageFromArray(movingMask.swapaxes(0,2))
#    moving_mask = sitk.Cast(moving_mask,sitk.sitkUInt8)
#    moving_mask.SetSpacing(spacing[0])
#    func.displayMontage(movingImg,5)
#    func.displayMontage(255*movingGT,5)
#    func.displayMontage(distMovingGT,5)
#    func.displayMontage(sitk.GetArrayFromImage(moving_image).swapaxes(0,2),5)
    zScaling = (moving_image.GetSize()[2]*moving_image.GetSpacing()[2])/(fixed_image.GetSize()[2]*fixed_image.GetSpacing()[2])
#    xyzScaling = tuple(map(lambda p,q,r,s:(p*q)/(r*s),moving_image.GetSize(),moving_image.GetSpacing(),fixed_image.GetSize(),fixed_image.GetSpacing()))
#    zScaling = 1
    scalingTform = sitk.AffineTransform(3)
    scalingTform.Scale((1,1,zScaling))
#    scalingTform.Scale(xyzScaling)
    resampleFlt = sitk.ResampleImageFilter()
    resampleFlt.SetTransform(scalingTform)
    resampleFlt.SetInterpolator(sitk.sitkLinear)
    resampleFlt.SetOutputOrigin(moving_image.GetOrigin())
    resampleFlt.SetOutputDirection(moving_image.GetDirection())
    resampleFlt.SetOutputPixelType(moving_image.GetPixelIDValue())
#    resampleFlt.SetOutputSpacing((fixed_image.GetSpacing()[0],fixed_image.GetSpacing()[1],fixed_image.GetSpacing()[2]))
    resampleFlt.SetOutputSpacing((moving_image.GetSpacing()[0],moving_image.GetSpacing()[1],fixed_image.GetSpacing()[2]))
#    resampleFlt.SetOutputSpacing(moving_image.GetSpacing())
#    resampleFlt.SetSize((fixed_image.GetSize()[0],fixed_image.GetSize()[1],fixed_image.GetSize()[2]))
    resampleFlt.SetSize((moving_image.GetSize()[0],moving_image.GetSize()[1],fixed_image.GetSize()[2]))
    moving_image = resampleFlt.Execute(moving_image)
    resampleFlt.SetOutputPixelType(moving_GT.GetPixelIDValue())
    moving_GT = resampleFlt.Execute(moving_GT)
#    movingGT1 = sitk.GetArrayFromImage(moving_GT1).swapaxes(0,2)
#    func.displayMontage(sitk.GetArrayFromImage(moving_image).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(moving_image1).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(fixed_image).swapaxes(0,2),5)
    
#    (final_transform,optimized_transform,init_transform) = findOptimumTform(movingGT_dist,fixedGT_dist,[],[],verbose=True)
    #(final_transform,optimized_transform,init_transform) = findOptimumTform(moving_GT,fixed_GT,[],[],verbose=True)
    (final_transform,optimized_transform,init_transform) = findOptimumTform(moving_image,fixed_image,[],[],verbose=True)
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, fixed_image.GetPixelIDValue())
    moving_resampledGT = sitk.Resample(moving_GT, fixed_GT, final_transform, sitk.sitkLinear, 0.0, fixed_GT.GetPixelIDValue())
    moving_initial = sitk.Resample(moving_image, fixed_image, init_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue())
#    func.displayMontageRGB(fixedImg,sitk.GetArrayFromImage(moving_initial).swapaxes(0,2),5)
#    func.displayMontageRGB(fixedImg,sitk.GetArrayFromImage(moving_resampled).swapaxes(0,2),5)
#    func.displayMontageRGB(fixedGT,255*(sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0,2)>127),5)
#    func.displayMontageRGB(fixedGT,sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(moving_resampled).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0,2),5)

    resampleFlt.SetSize((sitkMovingVol.GetSize()[0],sitkMovingVol.GetSize()[1],sitkFixedVol.GetSize()[2]))
    sitkMovingVol = resampleFlt.Execute(sitkMovingVol)
    sitkMovingGT = resampleFlt.Execute(sitkMovingGT)
    startInd2N = scalingTform.TransformPoint(startInd2)
#    func.displayMontage(sitk.GetArrayFromImage(sitkMovingVol).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(sitkMovingVol1).swapaxes(0,2),5)
#    func.displayMontage(sitk.GetArrayFromImage(sitkFixedVol).swapaxes(0,2),5)
    
    fixedParams = init_transform.GetFixedParameters()
    params = init_transform.GetParameters()
    #fixedParams = tuple(map(lambda x,y,z,s: x+(y-z)*s, fixedParams, tuple(startInd1), tuple(startInd2), (1.5625,1.5625,6)))
    fixedParams = tuple(map(lambda x,y,s: x+y*s, fixedParams, tuple(startInd1), tuple(spacingFixed)))
    tx = params[3]+(startInd2[0]*spacingMoving[0]-startInd1[0]*spacingFixed[0])
    ty = params[4]+(startInd2[1]*spacingMoving[1]-startInd1[1]*spacingFixed[1])
    tz = params[5]+(startInd2[2]*spacingMoving[2]-startInd1[2]*spacingFixed[2])
    
    init_transformNew = sitk.Euler3DTransform(fixedParams,params[0],params[1],params[2],(tx,ty,tz))
    affineParams = optimized_transform.GetParameters()
    optimized_transformNew = sitk.AffineTransform(affineParams[0:9],affineParams[9:],fixedParams)
    fullImg_transform = sitk.Transform(optimized_transformNew)
    fullImg_transform.AddTransform(init_transformNew)
        
    sitkRSVol = sitk.Resample(sitkMovingVol, sitkFixedVol, fullImg_transform, sitk.sitkLinear, 0.0, sitkFixedVol.GetPixelIDValue())
#    sitkMovingVolRS = sitk.Resample(sitkMovingVol,sitkFixedVol,sitk.Euler3DTransform(),sitk.sitkLinear,0.0,sitkFixedVol.GetPixelIDValue())
    sitkRSGT = sitk.Resample(sitkMovingGT, sitkFixedGT, fullImg_transform, sitk.sitkLinear, 0.0, sitkFixedGT.GetPixelIDValue())
#    func.displayMontageRGB(sitk.GetArrayFromImage(sitkFixedVol).swapaxes(0,2),sitk.GetArrayFromImage(sitkRSVol).swapaxes(0,2),5)
#    func.displayMontageRGB(sitk.GetArrayFromImage(sitkFixedGT).swapaxes(0,2),sitk.GetArrayFromImage(sitkRSGT).swapaxes(0,2),5)
#    func.displayMontageRGB(sitk.GetArrayFromImage(sitkFixedVol).swapaxes(0,2),sitk.GetArrayFromImage(sitkMovingVolRS).swapaxes(0,2),5)
    
#    reader = vtk.vtkXMLPolyDataReader()
#    exec("path = \"Meshes\mesh"+str(ind)+".vtp\"")
#    reader.SetFileName(path)
#    reader.Update()
#    mesh = reader.GetOutput()
#    points,poly = func.getPointsPoly(mesh)
#    tPoints = np.zeros(points.shape)
#    invFinalTform = fullImg_transform.GetInverse()
#    for k in range(points.shape[0]):
#        tPoints[k,:] = invFinalTform.TransformPoint(points[k,:])
#    
#    vtkTformPoints = vtk.vtkPoints()
#    vtkCell = vtk.vtkCellArray()
#    for k in range(tPoints.shape[0]):
#        vtkTformPoints.InsertPoint(k,tPoints[k,:])
#    for k in range(len(poly)):
#        vtkCell.InsertNextCell(3,poly[k])
#    tformMesh = vtk.vtkPolyData()
#    tformMesh.SetPoints(vtkTformPoints)
#    tformMesh.SetPolys(vtkCell)
##    tformGT = func.cutMesh(tformMesh,RSVMC50.shape,(1.5625,1.5625,6))
#    tformGT = func.cutMesh(tformMesh,avgVol.shape,(1.5625,1.5625,6))
#    
#    #sitkRSVol = sitk.Resample(sitkVol, sitkAvgVol, initial_transform.GetInverse(), sitk.sitkLinear, 0.0, sitkAvgVol.GetPixelIDValue())
#    #func.displayMontage(sitk.GetArrayFromImage(sitkRSVol).swapaxes(0,2),5)
#    #func.displayMontage(255*RSVMC50,5)
##    func.displayMontage(originalVol,5)
##    func.displayMontageRGB(255*RSVMC50,sitk.GetArrayFromImage(sitkRSVol).swapaxes(0,2),5)
#    #func.displayMontageRGB(RSGTMC50,sitk.GetArrayFromImage(sitkRSGT).swapaxes(0,2),5)
##    func.displayMontageRGB(RSGTMC50,tformGT,5)
#
#    func.displayMontageRGB(avgVol,sitk.GetArrayFromImage(sitkRSVol).swapaxes(0,2),5)
#    func.displayMontageRGB(avgGT,tformGT,5)
    
    exec("tformVol"+str(ind)+" = sitk.GetArrayFromImage(sitkRSVol).swapaxes(0,2)")
    exec("tformGT"+str(ind)+" = sitk.GetArrayFromImage(sitkRSGT).swapaxes(0,2)")
#    exec("tformGT"+str(ind)+" = tformGT")
    exec("dictVar.update({'tformVol"+str(ind)+"' : tformVol"+str(ind)+"})")
    exec("dictVar.update({'tformGT"+str(ind)+"' : tformGT"+str(ind)+"})")
    spacing = fixed_image.GetSpacing()
    dictVar.update({'spacing':spacing})
    dictVar.update({'IOP':IOPFixed})
#sio.savemat('N:\\ShusilDangi\\RVSC\\TrainingSetMat\\trainingSet.mat',dictVar)

#100(S),96(R),86(R),83(T),81(T,R,OK),72(T,OK),70(T),48(Sh),47(S,T),41(T,OK,Badmesh),38(T,Badmesh,OK),34(S),31(T),27(S),25(S,OK),
#21(S,OK),20(T,OK),19(T,OK),16(T),13(T),8(T),7(S),4(T,OK),1(T,OK)
# 66(T),36(T),