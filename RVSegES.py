# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:16:58 2016

@author: sxd7257
"""

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "gco_python-master"))
from pygco import cut_from_graph
import numpy as np
import myFunctions as func
import scipy.io as sio
import SimpleITK as sitk
# import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
# from tables import *
import time
from sklearn import mixture
import skimage.measure as measure
import skimage.morphology as skmorph
import skimage.segmentation as segment
import skimage.filters as filt
import math
import sklearn.metrics as metric
import nibabel as nib


def command_iteration(method):
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                           method.GetMetricValue(),
                                           method.GetOptimizerPosition()))


def findOptimumTform(moving_image, fixed_image, movingMask=[], fixedMask=[], verbose=True,
                     initializeCenteredTform=True):
    if initializeCenteredTform == True:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.MOMENTS)
    else:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)

    optimized_transform = sitk.AffineTransform(3)

    registration_method = sitk.ImageRegistrationMethod()

    if (len(movingMask) > 0):
        registration_method.SetMetricMovingMask(movingMask)

    if (len(fixedMask) > 0):
        registration_method.SetMetricFixedMask(fixedMask)

    # similarity metric settings
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsAmoeba(simplexDelta=1.0, numberOfIterations=1000,
                                             parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4,
                                             withRestarts=True)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 5, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # don't optimize in-place, we would possibly like to run this cell multiple times
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)
    registration_method.SetOptimizerWeights((1.0, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0))

    # connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    if (verbose):
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    registration_method.Execute(fixed_image, moving_image)
    final_transform = sitk.Transform(optimized_transform)
    final_transform.AddTransform(initial_transform)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return (final_transform, optimized_transform, initial_transform)


def findOptimumSliceTform(moving_image, fixed_image, movingMask=[], fixedMask=[], verbose=True):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler2DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    optimized_transform = sitk.AffineTransform(2)

    registration_method = sitk.ImageRegistrationMethod()

    if (len(movingMask) > 0):
        registration_method.SetMetricMovingMask(movingMask)

    if (len(fixedMask) > 0):
        registration_method.SetMetricFixedMask(fixedMask)

    # similarity metric settings
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricSamplingStrategy(registration_method.NONE)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # optimizer settings
    registration_method.SetOptimizerAsAmoeba(simplexDelta=1.0, numberOfIterations=1000,
                                             parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4,
                                             withRestarts=True)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # setup for the multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3,2,1])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # don't optimize in-place, we would possibly like to run this cell multiple times
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform)
    registration_method.SetOptimizerWeights((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))

    # connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))
    if (verbose):
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))

    registration_method.Execute(fixed_image, moving_image)
    final_transform = sitk.Transform(optimized_transform)
    final_transform.AddTransform(initial_transform)
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return (final_transform, optimized_transform, initial_transform)


def vector_norm(data, axis=None, out=None):
    # Return length, i.e. Euclidean norm, of ndarray along axis.
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
    """
    Return angle between vectors.
    If directed is False, the input vectors are interpreted as undirected axes,
    i.e. the maximum angle is pi/2.
    """
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    dot = np.sum(v0 * v1, axis=axis)
    dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
    return np.arccos(dot if directed else np.fabs(dot))


def findRotation(IOPMoving, IOPFixed):
    axisMoving = np.cross(IOPMoving[0:3], IOPMoving[3:6])
    angle = angle_between_vectors(IOPFixed[0:3], IOPMoving[0:3], directed=False)
    rotation1 = sitk.VersorTransform(tuple(axisMoving), angle)
    rotation2 = sitk.VersorTransform(tuple(axisMoving), -angle)
    tPoint1 = rotation1.TransformPoint(IOPFixed[0:3])
    tPoint2 = rotation2.TransformPoint(IOPFixed[0:3])
    if (np.sum(np.abs(np.asarray(tPoint1) - IOPMoving[0:3])) < np.sum(np.abs(np.asarray(tPoint2) - IOPMoving[0:3]))):
        rotation = sitk.VersorTransform((0, 0, 1), -angle)
    else:
        rotation = sitk.VersorTransform((0, 0, 1), angle)
    return rotation


def refineProbMap(segBP, probBP, probBPOriginal, ROIthresh):
    bpBinary = probBP > ROIthresh
    endoEdge = segment.find_boundaries(bpBinary.astype(int), connectivity=2, mode='outer', background=0)
    endo = morph.binary_fill_holes(endoEdge)
    # func.displayMontageRGB(vol,255*endoEdge)

    distEndo = morph.distance_transform_edt(np.logical_not(endoEdge), sampling=(1, 1), return_distances=True,
                                            return_indices=False)
    distEndo[endo > 0] = -distEndo[endo > 0]
    # func.displayMontage(distEndo)
    moving_Img = sitk.GetImageFromArray(distEndo.swapaxes(0, 1))
    moving_Img = sitk.Cast(moving_Img, sitk.sitkFloat32)
    moving_gt = sitk.GetImageFromArray(probBP.swapaxes(0, 1))
    moving_gt = sitk.Cast(moving_gt, sitk.sitkFloat32)
    moving_gt2 = sitk.GetImageFromArray(probBPOriginal.swapaxes(0, 1))
    moving_gt2 = sitk.Cast(moving_gt2, sitk.sitkFloat32)

    edgeBP = segment.find_boundaries(segBP.astype(int), connectivity=2, mode='outer', background=0)
    distBP = morph.distance_transform_edt(np.logical_not(edgeBP), sampling=(1, 1), return_distances=True,
                                          return_indices=False)
    distBP[segBP > 0] = -distBP[segBP > 0]
    # func.displayMontage(distBP)
    fixed_Img = sitk.GetImageFromArray(distBP.swapaxes(0, 1))
    fixed_Img = sitk.Cast(fixed_Img, sitk.sitkFloat32)

    fixed_mask = sitk.GetImageFromArray(255 * segBP.swapaxes(0, 1))
    fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)

    (final_tForm, optimized_tForm, init_tForm) = findOptimumSliceTform(moving_Img, fixed_Img, [], [], verbose=False)
    moving_RSgt = sitk.Resample(moving_gt, fixed_Img, final_tForm, sitk.sitkLinear, 0.0, fixed_Img.GetPixelIDValue())
    moving_RSgt2 = sitk.Resample(moving_gt2, fixed_Img, final_tForm, sitk.sitkLinear, 0.0, fixed_Img.GetPixelIDValue())
    # movingS_initial = sitk.Resample(moving_Img, fixed_Img, init_tForm, sitk.sitkLinear, 0.0, fixed_image.GetPixelIDValue())
    # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(movingS_initial).swapaxes(0,1))
    # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(moving_Img).swapaxes(0,1))
    # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(moving_RS).swapaxes(0,1))
    # func.displayMontageRGB(sitk.GetArrayFromImage(moving_gt).swapaxes(0,1),sitk.GetArrayFromImage(moving_RSgt).swapaxes(0,1))
    # func.displayMontageRGB(vol,255*(sitk.GetArrayFromImage(moving_gt).swapaxes(0,1)>0.5))
    # func.displayMontageRGB(vol,255*(sitk.GetArrayFromImage(moving_RSgt).swapaxes(0,1)>0.5))
    probBPRefined = sitk.GetArrayFromImage(moving_RSgt).swapaxes(0, 1)
    probBPRefinedOriginal = sitk.GetArrayFromImage(moving_RSgt2).swapaxes(0, 1)

    return (probBPRefined, probBPRefinedOriginal)


def selectCorrectBP(bp, endo, nearbySeg, idx):
    # Find connected component closeset to the endo
    bpMask1 = func.findBPSegment(bp, endo)
    bpMask2 = np.zeros(bpMask1.shape)
    if (np.sum(nearbySeg) > 0):
        # Find Connected Component closest to the BP Segmentation of nearby slice
        bpMask2 = func.findBPSegment(bp, nearbySeg)
    bpMask = np.logical_or(bpMask1, bpMask2)

    # Lump the two connected components into one
    _, nLabels = measure.label(bpMask, background=0, return_num=True, connectivity=1)
    r = 1
    while (nLabels > 1):
        bpMask = skmorph.closing(bpMask, skmorph.disk(r))
        _, nLabels = measure.label(bpMask, background=0, return_num=True, connectivity=1)
        r += 1

    if (np.sum(bpMask) > np.sum(nearbySeg) and idx < 0):
        # Initial BP oversegmented
        bpMask = np.copy(bpMask2)
    print('')
    return bpMask


def findThreshGMM(endoVol, endo):
    gmmInit = mixture.GaussianMixture(n_components=3)
    gmmInit.fit(endoVol[endo > 0].reshape(-1, 1))
    clustrs = gmmInit.predict(endoVol[endo > 0].reshape(-1, 1))
    (x, y) = np.nonzero(endo)
    initSeg = np.empty(endo.shape)
    initSeg[:] = np.NAN
    initSeg[x, y] = clustrs
    sortInd = np.argsort(gmmInit.means_, axis=0)
    bpGMM = initSeg == sortInd[2][0]
    threshGMM = np.min(endoVol[bpGMM > 0].ravel())
    return threshGMM


def evalDistances(gt, seg, spacing):
    gtPad = np.pad(gt, pad_width=2, mode='constant')
    segPad = np.pad(seg, pad_width=2, mode='constant')
    gtEdge = segment.find_boundaries(gtPad.astype(int), connectivity=2, mode='inner', background=0)
    segEdge = segment.find_boundaries(segPad.astype(int), connectivity=2, mode='inner', background=0)
    (gtX, gtY) = np.nonzero(gtEdge)
    ptsGT = np.vstack((gtX * spacing[0], gtY * spacing[1])).T
    (segX, segY) = np.nonzero(segEdge)
    ptsSeg = np.vstack((segX * spacing[0], segY * spacing[1])).T
    (_, dist1) = metric.pairwise_distances_argmin_min(ptsGT, ptsSeg)
    (_, dist2) = metric.pairwise_distances_argmin_min(ptsSeg, ptsGT)
    MAD = (np.mean(dist1) + np.mean(dist2))
    Hausdorff = max([np.max(dist1), np.max(dist2)])
    return (MAD, Hausdorff)


# Read the Average intensity and probability atlas
matContents = sio.loadmat('/Volumes/SAMSUNGUSB/RVSC/TrainingSetMat/avgVolGTES.mat')
keys = matContents.keys()
for i in keys:
    exec (i + "= matContents['" + i + "']")
spacingAvgVol = spacing[0]

# Read the LVRV mask obtained from KFir's ROI method
masks = sio.loadmat('/Volumes/SAMSUNGUSB/RVSC/TrainingSetMat/LvRvROIRVSC.mat')

# Histogram Matching Filter
histFilter = sitk.HistogramMatchingImageFilter()
histFilter.SetNumberOfHistogramLevels(256)
histFilter.SetNumberOfMatchPoints(5)
histFilter.SetThresholdAtMeanIntensity(False)

sitkMovingVol = sitk.GetImageFromArray(avgVol.swapaxes(0, 2))
sitkMovingVol = sitk.Cast(sitkMovingVol, sitk.sitkFloat32)
sitkMovingVol.SetSpacing(spacingAvgVol)
sitkMovingGT = sitk.GetImageFromArray(avgGT.swapaxes(0, 2))
sitkMovingGT = sitk.Cast(sitkMovingGT, sitk.sitkFloat32)
sitkMovingGT.SetSpacing(spacingAvgVol)
IOPMoving = IOP[0]

# Moving Volume
indNZ = np.nonzero(avgGT)
startInd2 = (np.min(indNZ, axis=1)).astype(int)
endInd2 = (np.max(indNZ, axis=1)).astype(int) + 1
movingImg = func.normalizeSlices(avgVol[startInd2[0]:int(1.25 * endInd2[0]), startInd2[1]:endInd2[1], :], 0, 100)
bSlice = np.zeros((movingImg.shape[0], movingImg.shape[1]))
movingImg = np.dstack((bSlice, bSlice, movingImg, bSlice, bSlice))
movingGT = avgGT[startInd2[0]:int(1.25 * endInd2[0]), startInd2[1]:endInd2[1], :]
movingGT = np.dstack((bSlice, bSlice, movingGT, bSlice, bSlice))
moving_image = sitk.GetImageFromArray(movingImg.swapaxes(0, 2))
moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
moving_image.SetSpacing(spacingAvgVol)
moving_GT = sitk.GetImageFromArray(movingGT.swapaxes(0, 2))
moving_GT = sitk.Cast(moving_GT, sitk.sitkFloat32)
moving_GT.SetSpacing(spacingAvgVol)
# func.displayMontage(sitk.GetArrayFromImage(moving_image).swapaxes(0,2),5)
# func.displayMontage(movingGT,5)


# Fixed Volume
indices = range(0, 16)

execTime = np.zeros(16)

# diceFile = open('dice.csv','wb')
# wDice = csv.writer(diceFile,delimiter=',',quoting=csv.QUOTE_NONE)
# jaccardFile = open('jaccard.csv','wb')
# wJaccard = csv.writer(jaccardFile,delimiter=',',quoting=csv.QUOTE_NONE)
# madFile = open('mad.csv','wb')
# wMad = csv.writer(madFile,delimiter=',',quoting=csv.QUOTE_NONE)
# hdFile = open('hd.csv','wb')
# wHd = csv.writer(hdFile,delimiter=',',quoting=csv.QUOTE_NONE)
# itrFile = open('itr.csv','wb')
# wItr = csv.writer(itrFile,delimiter=',',quoting=csv.QUOTE_NONE)
# timeFile = open('time.csv','wb')
# wTime = csv.writer(timeFile,delimiter=',',quoting=csv.QUOTE_NONE)

# indices = [0,3,6,7,8,9]

for ind in indices:
    numOfIterations = ()
    startTime = time.time()
    fileName = 'rawData' + str(ind) + '.mat'
    matContents = sio.loadmat('/Volumes/SAMSUNGUSB/RVSC/TrainingSetMat/' + fileName)
    # matContents = sio.loadmat('N:\\ShusilDangi\\RVSC\\TestSetMat\\'+fileName)
    keys = matContents.keys()
    for i in keys:
        exec (i + "= matContents['" + i + "']")
    for k in range(1,bp.shape[3]):
        if(np.sum(bp[:,:,:,k])>0):
            esFrame = k
            break
    v = np.copy(func.stretchContrast(vol[:, :, :, esFrame], 0, 99))
    # func.displayMontage(v,5)
    spacingFixed = np.copy(spacing[0])
    sitkFixedVol = sitk.GetImageFromArray(v.swapaxes(0, 2))
    sitkFixedVol = sitk.Cast(sitkFixedVol, sitk.sitkFloat32)
    sitkFixedVol.SetSpacing(spacingFixed)
    sitkFixedGT = sitk.GetImageFromArray(bp[:, :, :, esFrame].swapaxes(0, 2))
    sitkFixedGT = sitk.Cast(sitkFixedGT, sitk.sitkFloat32)
    sitkFixedGT.SetSpacing(spacingFixed)
    # func.displayMontageRGB(v,255*bp[:,:,:,esFrame],5)
    exec ('mask = masks[\'lvRvMask' + str(ind) + '\']')
    maskVol = np.tile(mask, (v.shape[2], 1, 1))
    maskVol = np.transpose(maskVol, [1, 2, 0])
    sitkMaskVol = sitk.GetImageFromArray(maskVol.swapaxes(0, 2))
    sitkMaskVol = sitk.Cast(sitkMaskVol, sitk.sitkFloat32)
    sitkMaskVol.SetSpacing(spacingFixed)
    IOPFixed = IOP[0]

    # Adjust Rotation based on Image Orientation Patient Information
    rotation = findRotation(IOPFixed, IOPMoving)
    centerFixed = tuple(map(lambda x, y: x * y / 2, sitkFixedVol.GetSize(), sitkFixedVol.GetSpacing()))
    rotation.SetCenter(centerFixed)
    sitkFixedVol = sitk.Resample(sitkFixedVol, sitkFixedVol, rotation, sitk.sitkLinear, 0.0,
                                 sitkFixedVol.GetPixelIDValue())
    sitkFixedGT = sitk.Resample(sitkFixedGT, sitkFixedGT, rotation, sitk.sitkLinear, 0.0, sitkFixedGT.GetPixelIDValue())
    sitkMaskVol = sitk.Resample(sitkMaskVol, sitkMaskVol, rotation, sitk.sitkLinear, 0.0, sitkMaskVol.GetPixelIDValue())
    rotVol = sitk.GetArrayFromImage(sitkFixedVol).swapaxes(0, 2)
    rotGT = sitk.GetArrayFromImage(sitkFixedGT).swapaxes(0, 2)
    rotMask = sitk.GetArrayFromImage(sitkMaskVol).swapaxes(0, 2)

    indNZ = np.nonzero(rotMask)
    startInd1 = (np.min(indNZ, axis=1)).astype(int)
    endInd1 = (np.max(indNZ, axis=1)).astype(int) + 1
    endInd1[2] = min(endInd1[2], v.shape[2])

    # Set ROI for z-axis using the provided ground-truth (it would be a manual input for current algorithm)
    indApexBase = np.nonzero(bp[:, :, :, esFrame])
    startIndApex = np.min(indApexBase, axis=1)
    endIndBase = np.max(indApexBase, axis=1) + 1
    cropVol = func.normalizeSlices(rotVol[startInd1[0]:endInd1[0], startInd1[1]:endInd1[1], :], 0, 100)
    bSlice = np.zeros((cropVol.shape[0], cropVol.shape[1]))
    cropVol = np.dstack((bSlice, bSlice, cropVol, bSlice, bSlice))

    # Set the cropped and rotated fixed volume, adding two blank slices in both ends to facilitate registration
    fixedImg = cropVol
    fixedGT = rotGT[startInd1[0]:endInd1[0], startInd1[1]:endInd1[1], :]
    fixedGT = np.dstack((bSlice, bSlice, fixedGT, bSlice, bSlice))
    # func.displayMontageRGB(fixedImg,fixedGT,5)
    fixedMask = rotMask[startInd1[0]:endInd1[0], startInd1[1]:endInd1[1], :]
    fixed_image = sitk.GetImageFromArray(fixedImg.swapaxes(0, 2))
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    fixed_image.SetSpacing(spacing[0])
    fixed_GT = sitk.GetImageFromArray(fixedGT.swapaxes(0, 2))
    fixed_GT = sitk.Cast(fixed_GT, sitk.sitkFloat32)
    fixed_GT.SetSpacing(spacing[0])
    fixed_mask = sitk.GetImageFromArray(fixedMask.swapaxes(0, 2))
    fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)
    fixed_mask.SetSpacing(spacing[0])
    # func.displayMontage(fixedImg,5)
    # func.displayMontage(255*fixedGT,5)
    func.displayMontageRGB(fixedImg, 255 * fixedGT, 5)

    # Histogram Match moving image to fixed image
    fixed_image = histFilter.Execute(fixed_image, moving_image)

    # zScaling = ((moving_image.GetSize()[2])*moving_image.GetSpacing()[2])/(fixed_image.GetSize()[2]*fixed_image.GetSpacing()[2])
    zScaling = 1
    scalingTform = sitk.AffineTransform(3)
    scalingTform.Scale((1, 1, zScaling))
    resampleFlt = sitk.ResampleImageFilter()
    resampleFlt.SetTransform(scalingTform)
    resampleFlt.SetInterpolator(sitk.sitkLinear)
    resampleFlt.SetOutputOrigin(moving_image.GetOrigin())
    resampleFlt.SetOutputDirection(moving_image.GetDirection())
    resampleFlt.SetOutputPixelType(moving_image.GetPixelIDValue())
    resampleFlt.SetOutputSpacing(moving_image.GetSpacing())
    resampleFlt.SetSize((moving_image.GetSize()[0], moving_image.GetSize()[1], fixed_image.GetSize()[2]))
    moving_image1 = resampleFlt.Execute(moving_image)
    resampleFlt.SetOutputPixelType(moving_GT.GetPixelIDValue())
    moving_GT1 = resampleFlt.Execute(moving_GT)
    movingGT1 = sitk.GetArrayFromImage(moving_GT1).swapaxes(0, 2)

    # Run the 3D registration algorithm
    (final_transform, optimized_transform, init_transform) = findOptimumTform(moving_image1, fixed_image, [],
                                                                              fixed_mask, verbose=True)
    moving_resampled = sitk.Resample(moving_image1, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     fixed_image.GetPixelIDValue())
    moving_resampledGT = sitk.Resample(moving_GT1, fixed_GT, final_transform, sitk.sitkLinear, 0.0,
                                       fixed_GT.GetPixelIDValue())
    moving_initial = sitk.Resample(moving_image1, fixed_image, init_transform, sitk.sitkLinear, 0.0,
                                   fixed_image.GetPixelIDValue())

    # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_image).swapaxes(0,2),sitk.GetArrayFromImage(moving_resampled).swapaxes(0,2),5)
    func.displayMontageRGB(sitk.GetArrayFromImage(fixed_image).swapaxes(0, 2),
                           sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0, 2), 5)
    # func.displayMontageRGB(255*sitk.GetArrayFromImage(fixed_GT).swapaxes(0,2),sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0,2),5)

    indNZ = np.nonzero(fixedGT)
    startInd = (np.min(indNZ, axis=1)).astype(int)
    endInd = (np.max(indNZ, axis=1)).astype(int) + 1

    fixedImg = fixedImg[:, :, startInd[2]:endInd[2]]
    fixedGT = fixedGT[:, :, startInd[2]:endInd[2]]
    movingNew = sitk.GetArrayFromImage(moving_resampled).swapaxes(0, 2)
    movingNew = movingNew[:, :, startInd[2]:endInd[2]]
    movingNewGT = sitk.GetArrayFromImage(moving_resampledGT).swapaxes(0, 2)
    movingNewGT = movingNewGT[:, :, startInd[2]:endInd[2]]

    ###################################################################################################################
    ########################### Process slicewise ################################

    # This is the order individual slices are processed
    # First the mid-slice is processed followed by its corresponding apical and basal slices
    sliceIDs = range(fixedImg.shape[2])
    midInd = int(np.floor(len(sliceIDs) / 2))
    indOrder = []
    for i in range(midInd, -1, -1):
        indOrder.append(i)
        if (i != 0):
            indOrder.append(-i)
    if (np.all(fixedImg[:, :, indOrder[0]] == fixedImg[:, :, indOrder[1]])):
        indOrder.remove(-midInd)

    movingRefined = np.zeros(movingNew.shape)
    movingRefinedGT = np.zeros(movingNew.shape)

    bpVol = np.zeros(fixedImg.shape)
    bpProb = np.zeros(fixedImg.shape)
    bpProbOriginal = np.zeros(fixedImg.shape)
    bpVolNonEmpty = np.zeros(fixedImg.shape)

    gBP = mixture.GaussianMixture(n_components=1)
    gBG = mixture.GaussianMixture(n_components=2)

    midProbMap = movingNewGT[:, :, midInd] / movingNewGT[:, :, midInd].max()
    # func.displayMontage(midProbMap)

    # Initializations
    bpROI = np.zeros(fixedImg[:, :, 0].shape)

    for idx in indOrder:
        vol = fixedImg[:, :, idx].astype(int)
        volFlt = filt.median(fixedImg[:, :, idx] / fixedImg[:, :, idx].max(), skmorph.disk(3))
        volFlt = func.normalizeSlices(volFlt, 0, 100)
        # gt = fixedGT[:,:,idx]
        bgInt = []
        bpInt = []

        if (idx == midInd):
            nearbySlice = idx
        elif (idx < 0):
            nearbySlice = idx - 1
        else:
            nearbySlice = idx + 1
            ########################################################################################################################
            ######### Segment the Bloodpool ########################################################################################

        # Assign BP Probability from registered atlas
        # If BP probability for current slice empty, borrow from nearby slice
        if (np.sum(movingNewGT[:, :, idx]) != 0):
            probBP = movingNewGT[:, :, idx] / movingNewGT[:, :, idx].max()
            probBPOriginal = movingNewGT[:, :, idx]
            bpVolNonEmpty[:, :, idx] = movingNewGT[:, :, idx]
        else:
            probBP = bpVolNonEmpty[:, :, nearbySlice] / bpVolNonEmpty[:, :, nearbySlice].max()
            probBPOriginal = bpVolNonEmpty[:, :, nearbySlice]
            bpVolNonEmpty[:, :, idx] = bpVolNonEmpty[:, :, nearbySlice]
        # func.displayMontageRGB(vol,255*probBP)

        ROIthresh = np.mean(probBP[probBP > 0])
        ROI = fixedMask[:, :, 0] > 0
        ROI = skmorph.remove_small_objects(ROI, min_size=20, connectivity=1)

        # Select large endo ROI for basal slices
        if (idx >= 0):
            endo = probBP > 0.01
        else:
            endo = probBP > ROIthresh
        endo = np.logical_and(endo > 0, ROI)
        if (np.sum(bpVol[:, :, nearbySlice]) > 0):
            endo = np.logical_or(endo, bpVol[:, :, nearbySlice])

        endoVol = np.multiply(volFlt, endo)
        # func.displayMontage(endoVol)
        threshOtsu = filt.threshold_otsu(endoVol[endo > 0].ravel())

        bp = endoVol >= threshOtsu
        # func.displayMontageRGB(vol,255*bp)

        bpMask = selectCorrectBP(bp, endo, bpVol[:, :, nearbySlice], idx)
        # func.displayMontageRGB(vol,255*bpMask)

        bpMaskInitial = np.copy(bpMask)
        segBP = bpMask > 0
        segBP = morph.binary_fill_holes(segBP)
        # func.displayMontageRGB(vol,255*segBP)

        # Initialization
        params = np.array([0, 0, 0, 0, 0, 0])
        newParams = np.array([1, 0, 0, 1, 0, 0])
        itr = 0
        dd = 0.7
        print('')
        print('')
        print('PROCESSING SLICE : ', idx)
        dice = 0

        try:
            while (dice < 0.99 and itr < 10):
                itr += 1
                segBPOld = np.copy(segBP)

                indBP = np.nonzero(bpMask)
                indBPInv = np.nonzero(np.logical_xor(bpMask, ROI))
                # func.displayMontageRGB(vol,255*bpMask)
                # func.displayMontageRGB(vol,255*np.logical_xor(bpMask,ROI))
                for i in vol[indBP[0], indBP[1]]: bpInt.append(i)
                gBP.fit(np.asarray(bpInt).reshape(-1, 1))
                # bpSamples = gBP.sample(len(vol[indBP[0],indBP[1]].ravel()))
                # plt.figure()
                # plt.hist(vol[indBP[0],indBP[1]].ravel(),bins=np.arange(255),normed=True)
                # plt.hist(bpSamples,bins=np.arange(255),normed=True)

                for i in vol[indBPInv[0], indBPInv[1]]: bgInt.append(i)
                gBG.fit(np.asarray(bgInt).reshape(-1, 1))
                # bgSamples = gBG.sample(len(vol[indBPInv[0],indBPInv[1]].ravel()))
                # plt.figure()
                # plt.hist(vol[indBPInv[0],indBPInv[1]].ravel(),bins=np.arange(255),normed=True)
                # plt.hist(bgSamples,bins=np.arange(255),normed=True)

                xx = gBP.score_samples(np.arange(0, 256).reshape(-1, 1))
                yy = gBG.score_samples(np.arange(0, 256).reshape(-1, 1))
                # fig,ax = plt.subplots()
                # ax.plot(xx,'k--',label="BP loglikelihood")
                # ax.plot(yy,'b',label="BG loglikelihood")
                # legend = ax.legend(loc='upper right')

                # create unaries based on intensity likelihood
                unariesLLS = 10 * xx[vol]
                unariesLLS[bpMask > 0] = 100
                unariesLLT = 10 * yy[vol]
                # func.displayMontage(unariesLLS-unariesLLT,5)
                # func.displayMontage(unariesLLS,5)
                # func.displayMontage(unariesLLT,5)

                atlasGT = np.copy(np.multiply(probBP, ROI))
                atlasGT = atlasGT.astype(float) / atlasGT.max()
                # atlasGT = 1-atlasGT
                # func.displayMontage(atlasGT)
                # func.displayMontageRGB(vol,255*atlasGT)
                # func.displayMontageRGB(vol,1-atlasGT)

                # LL = unariesLLS-unariesLLT
                # LL[bpMask>0]=100
                # for visualization purposes
                # func.displayMontage(LL)

                # as we convert to int, we need to multipy to get sensible values
                unaries = np.stack([unariesLLS, unariesLLT], axis=-1).copy("C").astype(np.int32)
                # create potts pairwise
                pairwise = -np.eye(2, dtype=np.int32)

                # use the general graph algorithm
                # first, we construct the grid graph
                inds = np.arange(vol.size).reshape(vol.shape)
                horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]  # horizontal edges
                vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]  # vertical edges
                # depth = np.c_[inds[:,:,:-1].ravel(), inds[:,:,1:].ravel()]  # slice edges
                edges = np.vstack([horz, vert]).astype(np.int32)
                # eWeight1 = 50*(np.absolute(vol.ravel()[edges[:,0]]-vol.ravel()[edges[:,1]]).reshape(edges.shape[0],1))
                # eWeight2 = 100*(atlasGT.ravel()[edges[:,0]]+atlasGT.ravel()[edges[:,1]]).reshape(edges.shape[0],1)
                eWeight1 = 50 * np.exp(
                    -(255 - np.absolute(vol.ravel()[edges[:, 0]] - vol.ravel()[edges[:, 1]])) / 255.0).reshape(
                    edges.shape[0], 1)
                eWeight2 = 500 * np.exp(-10.0 * (atlasGT.ravel()[edges[:, 0]] + atlasGT.ravel()[edges[:, 1]])).reshape(
                    edges.shape[0], 1)
                # edges = np.hstack((edges,eWeight1)).astype(np.int32)
                # edges = np.hstack((edges,eWeight2)).astype(np.int32)
                edges = np.hstack((edges, eWeight1 + eWeight2)).astype(np.int32)

                # we flatten the unaries
                result_graph = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)

                seg = result_graph.reshape(vol.shape)
                # func.displayMontageRGB(vol,255*seg)

                segBP = selectCorrectBP(seg, bpMaskInitial, bpVol[:, :, nearbySlice], idx)
                segBP = morph.binary_fill_holes(segBP)
                r = 10
                padSegBP = np.pad(segBP, pad_width=r, mode='constant')  # ,constant_values=((0,0),(0,0)))
                padSegBP = skmorph.closing(padSegBP, skmorph.disk(r))
                segBP = padSegBP[r:-r, r:-r]
                # func.displayMontageRGB(vol,255*segBP)
                # print('Here')

                # If the BP area for apical slice higher than nearby basal slice, adjust the threshold
                if (np.sum(segBP) > np.sum(bpVol[:, :, nearbySlice]) and idx < 0):
                    # BP oversegmented, change BP threshold to threshGMM
                    endoVol2 = np.multiply(volFlt, segBP)
                    threshGMM = findThreshGMM(endoVol2, segBP)
                    bp = endoVol >= threshGMM
                    bpMask = selectCorrectBP(bp, endo, bpVol[:, :, nearbySlice], idx)
                    bpMaskInitial = np.copy(bpMask)
                    segBP = bpMask > 0

                # Limit the segmentation within the endo ROI obtained from probability map
                if (np.sum(bpVol) > 0):
                    extraSeg = 1 * segBP - 1 * np.logical_or((np.sum(bpVol, axis=2) > 0), endo)
                else:
                    extraSeg = 1 * segBP - 1 * endo
                extraSeg[extraSeg < 0] = 0
                # func.displayMontage(extraSeg)
                # func.displayMontageRGB(255*segBP,255*endo)

                if (np.sum(extraSeg) > 0.05 * np.sum(segBP)):
                    # If slices oversegmented
                    if (itr == 1):
                        # If first iteration, reinitialize bp segment using threshGMM
                        threshGMM = findThreshGMM(endoVol, endo)
                        bp = endoVol >= threshGMM
                        bpMask = selectCorrectBP(bp, endo, bpVol[:, :, nearbySlice], idx)
                        bpMaskInitial = np.copy(bpMask)
                        segBP = bpMask > 0
                    elif (np.sum(bpVol[:, :, nearbySlice]) > 0):
                        # If not a middle slice, restrict the BP using segmentation from the nearby slice
                        segBP = np.logical_and(segBP, bpVol[:, :, nearbySlice])
                    else:
                        # If middle slice, restrict the BP using atlasGT
                        segBP = np.logical_and(segBP, atlasGT > 0.5)

                segRaw = np.multiply(seg, segBP)
                segRawFilled = morph.binary_fill_holes(segRaw)
                # func.displayMontageRGB(vol,255*segRaw,5)

                (_, _, _, _, dice, _) = func.evalMetrics(segBP, segBPOld, np.ones(segBP.shape))
                if (dice < 0.5 and np.sum(segBP) > np.sum(segBPOld)):
                    # RV expanded to LV or outside, stop iteration
                    print("LARGE BP EXPANSION!!!")
                    segBP = np.copy(segBPOld)
                    probBPRefined = np.copy(probBP)
                    probBPRefinedOriginal = np.copy(probBPOriginal)
                    break

                # Refine the probability map based on current blood-pool segmentation
                bpBinary = probBP > ROIthresh
                endoEdge = segment.find_boundaries(bpBinary.astype(int), connectivity=2, mode='outer', background=0)
                endo = morph.binary_fill_holes(endoEdge)
                # func.displayMontageRGB(vol,255*endoEdge)

                distEndo = morph.distance_transform_edt(np.logical_not(endoEdge), sampling=(1, 1),
                                                        return_distances=True, return_indices=False)
                distEndo[endo > 0] = -distEndo[endo > 0]
                # func.displayMontage(distEndo)
                moving_Img = sitk.GetImageFromArray(distEndo.swapaxes(0, 1))
                moving_Img = sitk.Cast(moving_Img, sitk.sitkFloat32)
                moving_gt = sitk.GetImageFromArray(probBP.swapaxes(0, 1))
                moving_gt = sitk.Cast(moving_gt, sitk.sitkFloat32)
                moving_gt2 = sitk.GetImageFromArray(probBPOriginal.swapaxes(0, 1))
                moving_gt2 = sitk.Cast(moving_gt2, sitk.sitkFloat32)

                edgeBP = segment.find_boundaries(segBP.astype(int), connectivity=2, mode='outer', background=0)
                distBP = morph.distance_transform_edt(np.logical_not(edgeBP), sampling=(1, 1), return_distances=True,
                                                      return_indices=False)
                distBP[segBP > 0] = -distBP[segBP > 0]
                # func.displayMontage(distBP)
                fixed_Img = sitk.GetImageFromArray(distBP.swapaxes(0, 1))
                fixed_Img = sitk.Cast(fixed_Img, sitk.sitkFloat32)

                fixed_mask = sitk.GetImageFromArray(255 * segBP.swapaxes(0, 1))
                fixed_mask = sitk.Cast(fixed_mask, sitk.sitkUInt8)

                (final_tForm, optimized_tForm, init_tForm) = findOptimumSliceTform(moving_Img, fixed_Img, [], [],
                                                                                   verbose=False)
                moving_RS = sitk.Resample(moving_Img, fixed_Img, final_tForm, sitk.sitkLinear, 0.0,
                                          fixed_Img.GetPixelIDValue())
                moving_RSgt = sitk.Resample(moving_gt, fixed_Img, final_tForm, sitk.sitkLinear, 0.0,
                                            fixed_Img.GetPixelIDValue())
                moving_RSgt2 = sitk.Resample(moving_gt2, fixed_Img, final_tForm, sitk.sitkLinear, 0.0,
                                             fixed_Img.GetPixelIDValue())
                movingS_initial = sitk.Resample(moving_Img, fixed_Img, init_tForm, sitk.sitkLinear, 0.0,
                                                fixed_image.GetPixelIDValue())
                # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(movingS_initial).swapaxes(0,1))
                # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(moving_Img).swapaxes(0,1))
                # func.displayMontageRGB(sitk.GetArrayFromImage(fixed_Img).swapaxes(0,1),sitk.GetArrayFromImage(moving_RS).swapaxes(0,1))
                # func.displayMontageRGB(sitk.GetArrayFromImage(moving_gt).swapaxes(0,1),sitk.GetArrayFromImage(moving_RSgt).swapaxes(0,1))
                # func.displayMontageRGB(vol,255*(sitk.GetArrayFromImage(moving_gt).swapaxes(0,1)>0.5))
                # func.displayMontageRGB(vol,255*(sitk.GetArrayFromImage(moving_RSgt).swapaxes(0,1)>0.5))
                probBPRefined = sitk.GetArrayFromImage(moving_RSgt).swapaxes(0, 1)
                probBPRefinedOriginal = sitk.GetArrayFromImage(moving_RSgt2).swapaxes(0, 1)
                # func.displayMontageRGB(vol,255*probBPRefined)
                # func.displayMontageRGB(vol,255*probBPRefinedOriginal)
                params = np.copy(newParams)
                newParams = np.asarray(optimized_tForm.GetParameters())
                print(itr, params, newParams)
                # func.displayMontage(segBP)
                # func.displayMontage(255*seg,5)
                # func.displayMontageRGB(255*gt,255*segBP)
                # func.displayMontageRGB(vol,255*segBP)

                # Setup for next iteration
                probBP = np.copy(probBPRefined)
                probBPOriginal = np.copy(probBPRefinedOriginal)
                ROI = probBP > 0.01
                bpMask = np.copy(segRawFilled)
                # func.displayMontageRGB(vol,255*segRawFilled)
                # func.displayMontageRGB(vol,255*segBP)

        except:
            print('WHILE LOOP!')
            continue
        #
        try:
            bpProb[:, :, idx] = probBPRefined
            bpProbOriginal[:, :, idx] = probBPRefinedOriginal
            bpVol[:, :, idx] = segBP
            numOfIterations = numOfIterations + (itr,)
        except:
            print('ASSIGNMENT')
            continue

    # Perform fourier smoothing
    bpVolSmooth = np.zeros(fixedImg.shape)
    for i in range(fixedImg.shape[2]):
        try:
            bpVolSmooth[:, :, i] = func.fourierSmoothing(bpVol[:, :, i], 8)
        except:
            continue

    func.displayMontageRGB(fixedImg, 255 * bpVol, 3)
    func.displayMontageRGB(fixedImg, 255 * bpVolSmooth, 3)
    # func.displayMontageRGB(255*fixedGT,255*bpVol,3)
    # func.displayMontageRGB(fixedImg,255*bpProb,5)
    # func.displayMontageRGB(fixedImg,255*bpProbOriginal,5)
    # func.displayMontageRGB(fixedImg,fixedGT,5)
    print('')

    # # Save image as Nifti
    # scalingArray = np.eye(4)
    # scalingArray[0,0] = spacingFixed[0]
    # scalingArray[1,1] = spacingFixed[1]
    # scalingArray[2,2] = spacingFixed[2]
    # niftiSegmented = nib.Nifti1Image(bpVolSmooth,scalingArray)
    # nib.save(niftiSegmented, 'SegmentedVolume.nii')
    # niftiGroundTruth = nib.Nifti1Image(fixedGT,scalingArray)
    # nib.save(niftiGroundTruth, 'GroundTruth.nii')
    # niftiVolume = nib.Nifti1Image(fixedImg,scalingArray)
    # nib.save(niftiVolume, 'Volume.nii')

################# Compute the results #########################################
# Moving Volume
#    execTime[ind] = time.time()-startTime
#    indNZ = np.nonzero(fixedGT)
#    startInd = (np.min(indNZ,axis=1)).astype(int)
#    endInd = (np.max(indNZ,axis=1)).astype(int)+1
#
#    gtImage = fixedGT[:,:,startInd[2]:endInd[2]]
#    segImage = bpVol[:,:,startInd[2]:endInd[2]]
#    func.displayMontageRGB(255*gtImage,255*segImage,5)
#
#    (dice,jaccard,_,_,_,_) = func.evaluateMetrics(gtImage,segImage,option='union')
#    (dice,jaccard,_,_,_,_) = func.evaluateMetrics(fixedGT,bpVol,option='union')
#    MAD = ()
#    HD = ()
#    for k in range(fixedGT.shape[2]):
#        mad,hd = evalDistances(fixedGT[:,:,k],bpVol[:,:,k],spacingFixed)
#        MAD = MAD+(mad,)
#        HD = HD+(hd,)
#    wDice.writerow(dice)
#    wJaccard.writerow(jaccard)
#    wMad.writerow(MAD)
#    wHd.writerow(HD)
#    wItr.writerow(numOfIterations)

# wTime.writerow(execTime)
# diceFile.close()
# jaccardFile.close()
# madFile.close()
# hdFile.close()
# itrFile.close()
# timeFile.close()
