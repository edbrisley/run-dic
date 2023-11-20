##IMPORT LIBRARIESMatchStereoViews
from tokenize import Double
import numpy as np
import numpy.linalg as la
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math
import cv2 as cv
from scipy.linalg import solve
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import statsmodels.api as sm
import patsy as ps
from patsy import dmatrices
from patsy import dmatrix
from numpy import loadtxt
import os
from configparser import ConfigParser
from fast_interp import interp2d
import copy
import glob
#SKIMAGE packages
from skimage import data
from skimage.util import img_as_float
from skimage.transform import warp, AffineTransform
from skimage.transform import FundamentalMatrixTransform, ProjectiveTransform
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
import skimage as sk
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches, corner_subpix)
from skimage.color import rgb2gray
import platform
import GQ_rosettacode_Python3 as gq

#------------------------------------------------------------------------------------
# Versioning the library

__version__= "0.3.1"

#
def LoadSettings():

    #load the configuration file containing the DIC settings
    configur = ConfigParser()
    configur.read('Settings.ini')
    #settings dictionary
    settings = dict()
    
    #measurement and discretisation type
    meas_type = configur.get('DICMode', 'MeasurementType')
    discretisation = configur.get('DICMode', 'Discretisation')
    #
    settings['MeasurementType'] = meas_type
    settings['Discretisation'] = discretisation
    
    if settings['MeasurementType'] == 'Planar':
        
        settings['PlanarImagesFolder'] = 'planar_images'
        #single image series
        settings['CameraNo'] = None
      
    if settings['MeasurementType'] in {'StereoPlanar', 'StereoCurved'}:
        
        #stereo camera-configuration image folders
        stereo_images_folder_0 = configur.get('Stereo', 'LeftCameraImageFolder')
        settings['StereoImagesFolder0'] = stereo_images_folder_0
        stereo_images_folder_1 = configur.get('Stereo', 'RightCameraImageFolder')
        settings['StereoImagesFolder1'] = stereo_images_folder_1

        #view-matching
        view_match = configur.get('ViewMatching', 'InitializeViewMatch')
        refine_view_match = configur.get('ViewMatching', 'RefineViewMatch')
        #
        settings['InitializeViewMatch'] = view_match
        settings['RefineViewMatch'] = refine_view_match
        
        #first image series to analyse is the left camera/camera 0
        settings['CameraNo'] = 0
        
        #internal and stereo calibration parameters folder
        cal_parameters_folder = configur.get('Stereo', 'CalibrationParametersFolder')
        settings['CalibrationParametersFolder'] = cal_parameters_folder
        
    if settings['Discretisation'] == 'Local':

        sub_size = configur.getint('Local', 'SubsetSize')
        step_size = configur.getint('Local', 'StepSize')
        sub_shape = configur.get('Local', 'SubsetShape')
        shape_function = configur.get('Local', 'DeformationModel')
        import_ = configur.get('Import', 'Local')

        settings['MeasurementPointsName'] = configur.get('Import', 'MeasurementPointsName')
        settings['SubsetSize'] = sub_size
        settings['DeformationModel'] = shape_function
        settings['StepSize'] = step_size
        settings['SubsetShape'] = sub_shape
        settings['ImportMeasurementPointsFromExternal'] = import_ 
 
    if settings['MeasurementType'] == 'Global':
        element_size = configur.getint('Global', 'ElementSize')
        element_type = configur.get('Global', 'ElementType')
        FE_regularization = configur.get('Global', 'RegularizationType')
        import_ = configur.get('Import', 'Global')
        quadrature_order = configur.get('Global', 'QuadratureOrder')

        settings['ElementSize'] = element_size
        settings['ElementType'] = element_type
        settings['RegularizationType'] = FE_regularization
        settings['ImportMeasurementPointsFromExternal'] = import_
        settings['QuadratureOrder'] = quadrature_order
        
        if import_ == 'Yes':
            settings['MeshParser'] = configur.get('Import', 'MeshParser')
    
    #preprocessing of image data
    GF_stddev = configur.getfloat('PreProcessing', 'GaussianFilterStdDev')
    GF_filtsize = configur.getint('PreProcessing', 'GaussianFilterSize')
    ROISelection = configur.get('PreProcessing', 'ROISelection')
    #
    settings['GaussianFilterStdDev'] = GF_stddev
    settings['GaussianFilterSize'] = GF_filtsize

    #reference strategy
    corr_refstrat = configur.get('ImageReferencing', 'CorrelationReferenceStrategy')
    datum_image = configur.getint('ImageReferencing', 'DatumImage')
    target_image = configur.getint('ImageReferencing', 'TargetImage')
    increment = configur.getint('ImageReferencing', 'Increment')
    reference_interpolation = configur.get('ImageReferencing', 'InterpolateReferenceImage')
    #
    settings['CorrelationReferenceStrategy'] = corr_refstrat
    settings['DatumImage'] = datum_image
    settings['TargetImage'] = target_image
    settings['Increment'] = increment
    settings['InterpolateReferenceImage'] = reference_interpolation

    #Initialization of optimisation routine
    settings['GNInitialization'] = configur.get('Optimisation', 'GNInitialization')

    #strain
    settings['VSGFilterOrder'] = configur.get('Strain', 'VSGFilterOrder')
    settings['VSGFilterSize'] = configur.getint('Strain', 'VSGFilterSize')
 
    print('\nInitial settings:\n')
    print(settings)

    return settings
#
def LoadStereoImages(settings):

    operating_system = platform.system()
    current_working_directory = os.getcwd()

    #STEREO DIC
    #load image sets
    if operating_system in {'Linux'}:
        image_folder_0 = '/{}'.format(settings['StereoImagesFolder0'])
        image_folder_1 = '/{}'.format(settings['StereoImagesFolder1'])
    #Windows
    else:
        image_folder_0 = '\{}'.format(settings['StereoImagesFolder0'])
        image_folder_1 = '\{}'.format(settings['StereoImagesFolder1'])
    #two image sets for stereo-dic
    image_location_0 = current_working_directory + image_folder_0
    image_location_1 = current_working_directory + image_folder_1
    
    #read images in directory
    image_set_0 = []
    image_set_1 = []
    #first image set
    for filename_0 in os.listdir(image_location_0):    
        image_set_0.append(filename_0)
    #second image set
    for filename_1 in os.listdir(image_location_1):    
        image_set_1.append(filename_1)

    print('\nStereo images loaded\n')

    print('image_set_0:')
    print(image_set_0)

    print('image_set_1:')
    print(image_set_1)

    return image_set_0, image_set_1
#
def LoadPlanarImages(settings):
    
    operating_system = platform.system()
    current_working_directory = os.getcwd()
    
    #Linux (need to add macOS option, same as Linux ?)
    if operating_system in {'Linux'}:
        image_folder = '/{}'.format(settings['PlanarImagesFolder'])
    #Windows
    else:
        image_folder = '\{}'.format(settings['PlanarImagesFolder'])
    #location of images for DIC measurement
    image_location = current_working_directory + image_folder
    #read images in directory
    image_set = []
    for filename in os.listdir(image_location):    
        image_set.append(filename)

    print('\nPlanar images loaded, image set:\n')
    print(image_set)
    return image_set
#
def SubsetRelativeCoordinates(settings):

    sub_size = int(settings['SubsetSize'])
    #relative/local coordinates of pixels within subset (the same for all subsets)
    [eta, xsi] = np.meshgrid( np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                              np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                              indexing='ij' )

    #flatten local coordinates to vectors
    xsi = np.array([xsi.flatten(order = 'F')]).T
    eta = np.array([eta.flatten(order = 'F')]).T

    return xsi, eta
#
def PlanarDICLocal(settings, image_set, ROI):

    #measurement point coordinates defined in datum image
    #xsi, eta: local coordinates defined within subset(s)
    if settings['CameraNo'] in {0, None}:

        #load measurement point from external file
        if settings['ImportMeasurementPointsFromExternal'] == 'Yes':
            measurement_points_name = settings['MeasurementPointsName']
            measurement_points = loadtxt('{}.csv'.format(measurement_points_name),
                                          delimiter = ",").astype('int')
           
        #define measurement points using the settings specified in the config file 
        if settings['ImportMeasurementPointsFromExternal'] == 'No':
            measurement_points = LocalMeasurementPoints(settings, ROI)
            #append the measurement points to the settings file
            #temporary line of code to store measurement points for later triangulation of displacement coefficients
            #np.savetxt("measurement_points.csv", measurement_points, delimiter = ",")
    
        xsi, eta = SubsetRelativeCoordinates(settings)
        settings['xsi'] = xsi
        settings['eta'] = eta
        settings['MeasurementPoints'] = measurement_points
        
        
        #save measurement point coordinates in datum image of left camera as reference
        #for later use in cross correlation
        if settings['MeasurementType'] in {'StereoCurved', 'StereoPlanar'}:
            settings['MeasurementPoints_L0'] = settings['MeasurementPoints']
            settings['ROI_L0'] = ROI

    if settings['CorrelationReferenceStrategy'] == 'Incremental':
        coefficients = IncrementalLocal(settings, image_set, ROI)

    elif settings['CorrelationReferenceStrategy'] == 'Absolute':
        coefficients = AbsoluteReferencing(settings, image_set, ROI)

    return coefficients
#
def StereoDICPlanar(settings, image_set_0, image_set_1, ROI_0):

    if settings['Discretisation'] == 'Local':
        #perform 2D-DIC on left camera image series
        coefficients_0 = PlanarDICLocal(settings, image_set_0, ROI_0)
        print('\nleft camera image series correlation complete')
        #right camera/ camera 1
        settings['CameraNo'] = 1
        ROI_1 = MatchStereoViews(settings, image_set_0, image_set_1)
        coefficients_1 = PlanarDICLocal(settings, image_set_1, ROI_1)
        print('\nright camera image series correlation complete')
        #displacements = TriangulateDisplacements(settings, coefficients_0, coefficients_1)
         
    else:
        #global
        pass

    #store output data in a dictionary
    data = dict()

    #store DIC displacement measurements to data dictionary
    if settings['DeformationModel'] == 'Affine':

        data['U_L'] = np.vstack((np.array([coefficients_0[0, :]]),
                                np.array([coefficients_0[3, :]]))).astype('double')
        
        data['U_R'] =  np.vstack((np.array([coefficients_1[0, :]]),
                                 np.array([coefficients_1[3, :]]))).astype('double')
    
    if settings['DeformationModel'] == 'Quadratic':

        data['U_L'] = np.vstack((np.array([coefficients_0[0, :]]),
                                np.array([coefficients_0[6, :]]))).astype('double')
        
        data['U_R'] =  np.vstack((np.array([coefficients_1[0, :]]),
                                 np.array([coefficients_1[6, :]]))).astype('double')

    return coefficients_0, coefficients_1, settings, data
#
def StereoDICCurved(settings):

    pass
#
def SIFTquick(F_ROI_padded, G):

    #keypoint matching/feature detection between reference images of from both cameras
    #create SIFT feature detector object
    sift = cv.xfeatures2d.SIFT_create()
    #find keypoints and descriptors in both images
    kp1, d1 = sift.detectAndCompute(F_ROI_padded, None)
    kp2, d2 = sift.detectAndCompute(G, None)

    #cast keypoints to numpy arrays for ease of use
    #F keypoints x and y coordinates
    N_xk1 = d1.shape[0]
    xk1 = np.zeros([N_xk1, 1])
    yk1 = np.zeros([N_xk1, 1])
    for i in range(0, N_xk1):
        xk1[i] =  kp1[i].pt[0]
        yk1[i] =  kp1[i].pt[1]
    Xk1 = np.vstack((xk1.T, yk1.T))

    #F1 keypoints x and y coordinates
    N_xk2 = d2.shape[0]
    xk2 = np.zeros([N_xk2, 1])
    yk2 = np.zeros([N_xk2, 1])
    for i in range(0, N_xk2):
        xk2[i] =  kp2[i].pt[0]
        yk2[i] =  kp2[i].pt[1]
    Xk2 = np.vstack((xk2.T, yk2.T))

    return [Xk1, Xk2], [kp1, kp2], [d1, d2]
#
def KeypointMatchRobust(Xk, d):

    #find matches between two images with knnmatch and identify good matches
    bf = cv.BFMatcher()
    #store the indices of the matches in F and G
    matches = bf.knnMatch(d[0], d[1], k=2)
    #only keep good matches using condition inisde for loop
    good_matches = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good_matches.append([m])

    #cast good match indices to numpy array
    n_matches = np.size(good_matches)
    goood_matches = np.zeros([n_matches, 2])
    for j in range(0, n_matches):
        #F descriptors
        goood_matches[j, 0] = good_matches[j][0].queryIdx
        #G descriptors
        goood_matches[j, 1] = good_matches[j][0].trainIdx

    #remove bad matches from feature detection results
    Xk1 = Xk[0][:, goood_matches[:,0].astype(int)]
    Xk2 = Xk[1][:, goood_matches[:,1].astype(int)]

    return [Xk1, Xk2]
#
def kNNSubCentres(settings, n_subsets, Xk, k = 8):

    #measurement point coordinates of reference image
    #image series from single camera (affine fit)
    if k == 8:
        Xo1 = settings['MeasurementPoints']

    #cross correlation of left and right cameras (projective transformation fit)
    if k == 16:
        Xo1 = settings['MeasurementPoints_L0']

    #identify K nearest keypoints(neighbours) to each subset centre, respectively
    kNN_indices = np.zeros([n_subsets, k])
    for i in range(0, n_subsets):
        #SSD between subset centre i and all keypoints
        diff = np.sqrt((Xo1[0,i] - Xk[0][0,:])**2 + (Xo1[1,i] - Xk[0][1,:])**2)
        idx = np.argpartition(diff, k)
        #k minimum SSD between subset centre i and all keypoints
        kNN_indices[i, :] = idx[:k]

    return kNN_indices
#
def AffineFitKeypointsRANSAC(settings, n_subsets, Xk, kNN_indices):

    #storage array for relative displacement between matched
    #measurement points in reference and deformed image
    U = np.zeros_like(settings['MeasurementPoints'])

    for i_subset in range(0, n_subsets):
        #measurement point coordinates in reference image
        xo1 = settings['MeasurementPoints'][0, i_subset]
        yo1 = settings['MeasurementPoints'][1, i_subset]

        #find kNN keypoint coordinates for current subset
        xkp1 = Xk[0][:, kNN_indices[i_subset, :].astype(int)].T
        xkp2 = Xk[1][:, kNN_indices[i_subset, :].astype(int)].T
        model_robust, inliers = ransac((xkp1, xkp2), AffineTransform, min_samples=3,
                                        residual_threshold=2, max_trials=100)
        
        #affine transformation homography coefficients
        h11 = model_robust.params[0][0] 
        h12 = model_robust.params[0][1]
        h13 = model_robust.params[0][2]
        h21 = model_robust.params[1][0]
        h22 = model_robust.params[1][1]
        h23 = model_robust.params[1][2]
        
        #coordinates of measurement points in the deformed image
        xo2 = h11*xo1 + h12*yo1 + h13
        yo2 = h21*xo1 + h22*yo1 + h23

        #relative displacement between measurement points in reference and
        #deformed image
        #(note that the coordinate systems of the reference and deformed images
        #are identical - this implies that we can find relative displacements
        #of the measurement points between two images using a single coordinate frame)

        U[0, i_subset], U[1, i_subset] =  [ np.floor(xo2).astype('int') - xo1,
                                            np.floor(yo2).astype('int') - yo1 ]
        
    return U[0, :], U[1, :]
#
def KPMatchMeasurementPoints(settings, n_subsets, F_ROI_padded, G):
    
    #keypoint coordinates, keypoint objects, descriptors
    Xk, kp, d = SIFTquick(F_ROI_padded, G)
    #robust matching of keypoints between images
    Xk = KeypointMatchRobust(Xk, d)
    #find k nearest keypoints (neighbours) to each subset centre
    kNN_indices = kNNSubCentres(settings, n_subsets, Xk)
    #find measurementpoints defined in reference image in the
    #deformed image and estimate deformation as displacements
    u0, v0 = AffineFitKeypointsRANSAC(settings, n_subsets, Xk, kNN_indices)

    return u0, v0
#
def ProjectiveFitKeypointsRANSAC(settings, n_subsets, Xk, kNN_indices):

    #storage array for relative displacement between matched
    #measurement points in reference and deformed image
    Xo2 = np.zeros_like(settings['MeasurementPoints'])

    for i_subset in range(0, n_subsets):
        #measurement point coordinates in reference image
        xo1 = settings['MeasurementPoints_L0'][0, i_subset]
        yo1 = settings['MeasurementPoints_L0'][1, i_subset]

        #find kNN keypoint coordinates for current subset
        xkp1 = Xk[0][:, kNN_indices[i_subset, :].astype(int)].T
        xkp2 = Xk[1][:, kNN_indices[i_subset, :].astype(int)].T
        model_robust, inliers = ransac((xkp1, xkp2), ProjectiveTransform, min_samples=4,
                                        residual_threshold=2, max_trials=100)
        
        #projective transformation homography coefficients
        h11 = model_robust.params[0][0] 
        h12 = model_robust.params[0][1]
        h13 = model_robust.params[0][2]
        h21 = model_robust.params[1][0]
        h22 = model_robust.params[1][1]
        h23 = model_robust.params[1][2]
        h31 = model_robust.params[2][0]
        h32 = model_robust.params[2][1]
        h33 = model_robust.params[2][2]

        #coordinates of measurement points in deformed image
        xo2 = (h11*xo1 + h12*yo1 + h13)/(h31*xo1 + h32*yo1 + h33)
        yo2 = (h21*xo1 + h22*yo1 + h23)/(h31*xo1 + h32*yo1 + h33)

        if (settings['InterpolateReferenceImage']) == 'No':

            xo2 = np.floor(xo2).astype('int')
            yo2 = np.floor(yo2).astype('int')

        Xo2[0, i_subset], Xo2[1, i_subset] =  [ xo2,
                                               yo2 ]

    #store measurement points for planar DIC on the right camera image series
    settings['MeasurementPoints_R0'] = Xo2
    settings['MeasurementPoints'] = Xo2
    #temporary line of code to store measurement points for later triangulation of displacement coefficients
    #np.savetxt("R_measurement_points.csv", Xo2, delimiter = ",")

    #xy bounds for ROI of datum image of right camera image series
    rx_min = np.min(Xo2[0, :]) - 2*settings['SubsetSize']
    rx_max = np.max(Xo2[0, :]) + 2*settings['SubsetSize']
    ry_min = np.min(Xo2[1, :]) - 2*settings['SubsetSize']
    ry_max = np.max(Xo2[1, :]) + 2*settings['SubsetSize']

    #ROI in right camera image series
    ROI_1 =  (np.floor(rx_min).astype('int'),
              np.floor(ry_min).astype('int'),
              np.floor(rx_max - rx_min).astype('int'),
              np.floor(ry_max - ry_min).astype('int'))

    return ROI_1
#
def MatchStereoViews(settings, image_set_0, image_set_1):

    #left camera datum image ROI
    ROI_L0 = settings['ROI_L0']
    
    #left and right camera datum images
    L0_8bit = settings['StereoImagesFolder0'] + '/' + image_set_0[settings['DatumImage']]
    R0_8bit = settings['StereoImagesFolder1'] + '/' + image_set_1[settings['DatumImage']] 
   
    L0_8bit = cv.imread(L0_8bit, 0)
    R0_8bit = cv.imread(R0_8bit, 0)

    #change storage to data dictionary
    settings['L0_8bit'] = L0_8bit
    settings['R0_8bit'] = R0_8bit
    
    L0_ROI_padded = np.zeros_like(L0_8bit)

    #pad the left camera reference image ROI with zeros
    L0_ROI_padded[int(ROI_L0[1]):int(ROI_L0[1]+ROI_L0[3]),
                  int(ROI_L0[0]):int(ROI_L0[0]+ROI_L0[2])] \
    = L0_8bit[int(ROI_L0[1]):int(ROI_L0[1]+ROI_L0[3]),
              int(ROI_L0[0]):int(ROI_L0[0]+ROI_L0[2])]

    #perform keypoint matching between L0 and R0, find the corresponding measurement
    #points in R0 that are defined in L0 by fitting a projective transformation model
    #between corresponding keypoints in the images around the measurement points in L0
    
    n_subsets = settings['MeasurementPoints'].shape[1]
    #keypoint coordinates, keypoint objects, descriptors
    #settings, F_ROI_padded, F, G
    Xk, kp, d = SIFTquick(L0_ROI_padded, R0_8bit)
    
    #robust matching of keypoints between images
    Xk = KeypointMatchRobust(Xk, d)
    #find k nearest keypoints (neighbours) to each subset centre
    kNN_indices = kNNSubCentres(settings, n_subsets, Xk, 16)
    #find measurement points defined in reference image in the
    #deformed image and estimate deformation as displacements
    ROI_1 = ProjectiveFitKeypointsRANSAC(settings, n_subsets, Xk, kNN_indices)

    if settings['RefineViewMatch'] == 'Yes':
        CrossCorrelateViews(settings, image_set_0, image_set_1)

    return ROI_1
#
def ReferenceImage(settings, image_set, i_image):

    #pre-blur filter parameters
    GF_filt_size = settings['GaussianFilterSize']
    GF_std_dev = settings['GaussianFilterStdDev']

    if settings['MeasurementType'] in {'StereoPlanar', 'StereoCurved'}:
        #left camera
        if settings['CameraNo'] == 0:
            F_8bit = settings['StereoImagesFolder0'] + '/' + image_set[i_image]
        #right camera
        else:
            F_8bit = settings['StereoImagesFolder1'] + '/' + image_set[i_image]  
    
    #2D dic
    else:
        F_8bit = settings['PlanarImagesFolder'] + '/' + image_set[i_image]

    print(F_8bit)

    F_8bit = cv.imread(F_8bit, 0)
    
    #normalise 8 bit image, (0,255) -> (0,1)
    F = cv.normalize(F_8bit.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #blur image with gaussian filter
    if GF_filt_size != 0 and GF_std_dev != 0:
        F = cv.GaussianBlur(F,
                            (GF_filt_size, GF_filt_size),
                            GF_std_dev)
                             
    #gradient of F (reference image) in xy directions
    delF = np.array(np.gradient(F), dtype= float)
    F_interpolated = FastInterpolation(F)
    delF_interpolated = [FastInterpolation(delF[0]), FastInterpolation(delF[1])]

    return F, F_8bit, F_interpolated, delF, delF_interpolated    
#
def DeformedImage(settings, image_set, i_image):

    #gaussian filter settings
    GF_filt_size = settings['GaussianFilterSize']
    GF_std_dev = settings['GaussianFilterStdDev']

    if settings['MeasurementType'] in {'StereoPlanar', 'StereoCurved'}:
        #left or right camera
        if settings['CameraNo'] == 0:
            G_8bit = settings['StereoImagesFolder0'] + '/' + image_set[i_image + settings['Increment']]
        else:
            G_8bit = settings['StereoImagesFolder1'] + '/' + image_set[i_image + settings['Increment']]
    
    #2D dic
    else:
        G_8bit = settings['PlanarImagesFolder'] + '/' + image_set[i_image]
    G_8bit = cv.imread(G_8bit, 0)
    
    #normalise 8 bit image, (0,255) -> (0,1)
    G = cv.normalize(G_8bit.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #blur image with gaussian filter
    if GF_filt_size != 0 and GF_std_dev != 0:
        G = cv.GaussianBlur(G,
                            (GF_filt_size, GF_filt_size),
                            GF_std_dev)

    G_interpolated = FastInterpolation(G)
    
    return G, G_8bit, G_interpolated
#
def ConvergenceParametersGlobal(settings):

    pass
#
def HessianLocal(settings, delf, xsi, eta):

    #flatten subset intensity gradients to vectors
    if settings['InterpolateReferenceImage'] == 'No':
        dfdx = np.array([delf[1].flatten(order = 'F')]).T
        dfdy = np.array([delf[0].flatten(order = 'F')]).T

    if settings['InterpolateReferenceImage'] == 'Yes':
        
        dfdx = delf[1]
        dfdy = delf[0]

    #Affine transformation
    if settings['DeformationModel'] == 'Affine': 
        
        Jacobian = np.array([ dfdx[:,0]*1,
                              dfdx[:,0]*xsi[:,0],
                              dfdx[:,0]*eta[:,0],
                              dfdy[:,0]*1,
                              dfdy[:,0]*xsi[:,0],
                              dfdy[:,0]*eta[:,0] ]).T                            
    
    elif settings['DeformationModel'] == 'Quadratic':
        
        Jacobian = np.array([ dfdx[:,0]*1,
                              dfdx[:,0]*xsi[:,0],
                              dfdx[:,0]*eta[:,0],
                              dfdx[:,0]*0.5*xsi[:,0]**2,
                              dfdx[:,0]*xsi[:,0]*eta[:,0],
                              dfdx[:,0]*0.5*eta[:,0]**2,
                              dfdy[:,0]*1,
                              dfdy[:,0]*xsi[:,0],
                              dfdy[:,0]*eta[:,0],
                              dfdy[:,0]*0.5*xsi[:, 0]**2,
                              dfdy[:,0]*xsi[:,0]*eta[:,0],
                              dfdy[:,0]*0.5*eta[:,0]**2 ]).T
                                
    Hessian = np.dot(Jacobian.T, Jacobian) 

    return Hessian, Jacobian
#
def InitialiseConvergenceParamtersLocal(settings):

    n_subsets = settings['MeasurementPoints'].shape[1]
    correlation_coefficients = np.zeros([1, n_subsets])
    exit_crits = np.zeros([1, n_subsets])
    n_iterations = np.zeros([1, n_subsets])

    return correlation_coefficients, exit_crits, n_iterations
#
def InitialiseDeformationModelSubsets(settings, n_subsets, F_8bit, G_8bit, ROI):
    
    #initial estimate of the deformation model
    #the initial model estimate contains only displacements: u0, v0
    #performed here for all the subsets in the ROI

    if settings['GNInitialization'] == 'PhaseCorrelation':

        u0, v0 = EstimateDisplacementsFourier

    if settings['GNInitialization'] == 'TinyDisplacements':

        u0 = np.zeros([1, settings['MeasurementPoints'].shape[1]])
        v0 = np.zeros([1, settings['MeasurementPoints'].shape[1]])

    if settings['GNInitialization'] == 'KeypointDetection':
        #pad the ROI intensity values with zeros
        F_ROI_padded = np.zeros_like(F_8bit)
        F_ROI_padded[int(ROI[1]):int(ROI[1]+ROI[3]),
                     int(ROI[0]):int(ROI[0]+ROI[2])] = F_8bit[int(ROI[1]):int(ROI[1]+ROI[3]),
                                                              int(ROI[0]):int(ROI[0]+ROI[2])]
        F_ROI_padded = F_ROI_padded.astype(np.uint8)

        u0, v0 = KPMatchMeasurementPoints(settings, n_subsets,
                                          F_ROI_padded, G_8bit)

    if settings['GNInitialization'] == 'HornSchunck':

        pass
    
    if settings['GNInitialization']  == 'LucasKanade':
        
        pass

    #the estimated displacements correspond to the 'zeroth' coefficients of
    #the deformation model
    if settings['DeformationModel'] == 'Affine':

        coeffs = np.zeros([6, n_subsets])
        coeffs[0,:], coeffs[3,:] = u0, v0
    #
    if settings['DeformationModel'] == 'Quadratic':

        coeffs = np.zeros([12, n_subsets])
        coeffs[0,:], coeffs[6,:] = u0, v0

    return coeffs
#
def InitialiseCoefficientUpdate(settings):

    if settings['DeformationModel'] == 'Affine':

        deltaP =  np.ones([6,1])

    elif settings['DeformationModel'] == 'Quadratic':
        
        deltaP = np.ones([12,1])

    return deltaP
#
def GaussNewtonLocal(settings, ROI, image_set, i_image, image_pair):

    sub_size = settings['SubsetSize']
    #number of intensity values in the subset/template
    
    #reference subset local/relative coordinates (same for all subsets)
    xsi, eta = settings['xsi'], settings['eta']
    CC, exit_crits, n_iterations = InitialiseConvergenceParamtersLocal(settings)
    n_subsets = CC.shape[1]

    #define reference and target images for current image pair
    #delF: dFdy = delF[0], dFdx = delF[1]
    F, F_8bit, F_interpolated, delF, delF_interpolated = ReferenceImage(settings, image_set,
                                                     i_image)
    G, G_8bit, G_interpolated = DeformedImage(settings, image_set, i_image)
    
    #deformation model coefficients initialisation (all subsets)
    #p is the vector of coefficients of the subset deformaton model
    p = InitialiseDeformationModelSubsets(settings, n_subsets, F_8bit, G_8bit, ROI)
    
    #loop through all subsets/measurement points, determine the model coefficients for
    #each subset independently
    for i_subset in range(0, n_subsets):
        
        #measurement point/subset centre coordinates for i'th/current subset
        xo, yo = [ settings['MeasurementPoints'][0][i_subset],
                   settings['MeasurementPoints'][1][i_subset] ]

        #intensity data for reference subset
        if settings['InterpolateReferenceImage'] == 'No':
            f, f_mean, f_tilde, delf = ReferenceSubset(F, delF, sub_size, xo, yo, settings)

        if settings['InterpolateReferenceImage'] == 'Yes':
            f, f_mean, f_tilde, delf = ReferenceSubset(F_interpolated, delF_interpolated,
                                                       sub_size, xo, yo, settings)

        #Hessian and Jacobian operators for GN optimization routine, derived from
        #reference subset intensity gradient data
        H, J = HessianLocal(settings, delf, xsi, eta)

        #initial estimate for the incremental update of the model coefficients
        delta_p_i = InitialiseCoefficientUpdate(settings)
        #model coefficients
        p_i = p[:, i_subset]

        #check the dimensions

        iteration = 0
        #perform GN optimisation routine
        while iteration < 100:
            
            #relative coordinates of intensity values within subset
            #of deformed image, based on current iteration of deformation model
            xsi_d, eta_d = DeformationModelSubset(settings, p_i, xsi, eta)
            #intensity data of deformed subset
            g, g_mean, g_tilde = DeformedSubset(G_interpolated, xsi_d, eta_d, xo, yo)
            #
            exit_criteria = ExitCriteriaLocal(settings, delta_p_i, 0.5*(sub_size-1))
            if exit_criteria < 1e-4:
                break 
            else:
                residual = f[:]-f_mean-f_tilde/g_tilde*(g[:]-g_mean)
                b = np.dot(J.T, residual)
                #Gauss-Newton update of deformation model coefficients (for current subset)
                #i.e solve the linear system H*dp = b for i'th subset
                delta_p_i = np.linalg.solve(-H, b) 
                p_i = CompositionalUpdate(settings, p_i, delta_p_i)

            iteration += 1
        #store convergence information for current subset
        #coefficients of deformation model, correlation criterion,
        #exit criterion, number of iterations
        CC[0, i_subset] = 1 - sum(((f[:]-f_mean)/f_tilde-(g[:]-g_mean)/g_tilde)**2)/2
        exit_crits[0, i_subset] = exit_criteria
        n_iterations[0, i_subset]  = iteration
        p[:, i_subset] = p_i.ravel()

    return p 
#
def CrossCorrelateViews(settings, image_set_0, image_set_1):

    sub_size = settings['SubsetSize']
    #number of intensity values in the subset/template
    
    #reference subset local/relative coordinates (same for all subsets)
    xsi, eta = settings['xsi'], settings['eta']
    CC, exit_crits, n_iterations = InitialiseConvergenceParamtersLocal(settings)
    n_subsets = CC.shape[1]

    #define reference and target images for current image pair
    #delF: dFdy = delF[0], dFdx = delF[1]
    settings['CameraNo'] = 0
    F, F_8bit, F_interpolated, delF, delF_interpolated = ReferenceImage(settings, image_set_0, 0)
    settings['CameraNo'] = 1
    G, G_8bit, G_interpolated = DeformedImage(settings, image_set_1, -1)
    
    #deformation model coefficients initialisation (all subsets)
    #p is the vector of coefficients of the subset deformaton model
    #initial estimate for the incremental update of the model coefficients
    if settings['DeformationModel'] == 'Affine':

        p = np.zeros([6, n_subsets])
        p[0, :] = settings['MeasurementPoints_R0'][0, :] - settings['MeasurementPoints_L0'][0, :]
        p[3, :] = settings['MeasurementPoints_R0'][1, :] - settings['MeasurementPoints_L0'][1, :]
    
    if settings['DeformationModel'] == 'Quadratic':
        p = np.zeros([12, n_subsets])
        p[0, :] = settings['MeasurementPoints_R0'][0, :] - settings['MeasurementPoints_L0'][0, :]
        p[6, :] = settings['MeasurementPoints_R0'][1, :] - settings['MeasurementPoints_L0'][1, :]

    Xo2 = np.zeros_like(settings['MeasurementPoints'])
    for i_subset in range(0, n_subsets):
        
        #measurement point/subset centre coordinates for i'th/current subset
        xo, yo = [ settings['MeasurementPoints_L0'][0][i_subset],
                   settings['MeasurementPoints_L0'][1][i_subset] ]

        #intensity data for reference subset
        if settings['InterpolateReferenceImage'] == 'No':
            f, f_mean, f_tilde, delf = ReferenceSubset(F, delF, sub_size, xo, yo, settings)

        if settings['InterpolateReferenceImage'] == 'Yes':
            f, f_mean, f_tilde, delf = ReferenceSubset(F_interpolated, delF_interpolated,
                                                       sub_size, xo, yo, settings)

        #Hessian and Jacobian operators for GN optimization routine, derived from
        #reference subset intensity gradient data
        H, J = HessianLocal(settings, delf, xsi, eta)

        #initial estimate for the incremental update of the model coefficients
        delta_p_i = InitialiseCoefficientUpdate(settings)

        #model coefficients
        p_i = p[:, i_subset]

        iteration = 0
        #perform GN optimisation routine
        while iteration < 100:
            
            #relative coordinates of intensity values within subset
            #of deformed image, based on current iteration of deformation model
            xsi_d, eta_d = DeformationModelSubset(settings, p_i, xsi, eta)
            #intensity data of deformed subset
            g, g_mean, g_tilde = DeformedSubset(G_interpolated, xsi_d, eta_d, xo, yo)
            #
            exit_criteria = ExitCriteriaLocal(settings, delta_p_i, 0.5*(sub_size-1))
            if exit_criteria < 1e-4:
                break 
            else:
                residual = f[:]-f_mean-f_tilde/g_tilde*(g[:]-g_mean)
                b = np.dot(J.T, residual)
                #Gauss-Newton update of deformation model coefficients (for current subset)
                #i.e solve the linear system H*dp = b for i'th subset
                delta_p_i = np.linalg.solve(-H, b) 
                p_i = CompositionalUpdate(settings, p_i, delta_p_i)

            iteration += 1
        #store convergence information for current subset
        #coefficients of deformation model, correlation criterion,
        #exit criterion, number of iterations
        CC[0, i_subset] = 1 - sum(((f[:]-f_mean)/f_tilde-(g[:]-g_mean)/g_tilde)**2)/2
        exit_crits[0, i_subset] = exit_criteria
        n_iterations[0, i_subset]  = iteration
        p[:, i_subset] = p_i.ravel()

        #check interpolation of the reference image
        if settings['InterpolateReferenceImage'] == 'No':
            
            #check the deformation model order
            if settings['DeformationModel'] == 'Affine':

                Xo2[0, i_subset], Xo2[1, i_subset] =  [ np.floor(xo - p[0, i_subset]).astype('int'),
                                                        np.floor(yo - p[3, i_subset]).astype('int') ]   
            else: 
                
                Xo2[0, i_subset], Xo2[1, i_subset] =  [ np.floor(xo - p[0, i_subset]).astype('int'),
                                                        np.floor(yo - p[6, i_subset]).astype('int') ]

        else:

            #check the deformation model order
            if settings['DeformationModel'] == 'Affine':
            
                Xo2[0, i_subset], Xo2[1, i_subset] =  [ xo - p[0, i_subset],
                                                        yo - p[3, i_subset] ]
                
            else:

                Xo2[0, i_subset], Xo2[1, i_subset] =  [ xo - p[0, i_subset],
                                                        yo - p[6, i_subset] ]   

    np.savetxt("R_measurement_points_CC.csv", CC, delimiter = ",")
    settings['MeasurementPoints_R0'] = Xo2
    settings['MeasurementPoints'] = Xo2
    ##np.savetxt("R_measurement_points.csv", Xo2, delimiter = ",")

    print('\nCross Correlation complete')

    pass

def CompositionalUpdate(settings, p, dp):

    #
    if settings['DeformationModel'] == 'Affine':
        #w of current estimate of SFPs
        #order of SFP's P[1]: 0,  1,  2,  3,  4,  5
        #                     u   ux  uy  v   vx  vy
        w_P = np.array([ [1+p[1],    p[2],    p[0]],
                         [ p[4],    1+p[5],   p[3]],
                         [  0,        0,        1 ] ]).astype('double')

        #w of current delta_p               
        w_dP = np.array([ [1+dp[1],     dp[2],    dp[0]],
                          [ dp[4],     1+dp[5],   dp[3]],
                          [   0,          0,        1  ] ]).astype('double')

        #p coefficients compositional update matrix                
        up = np.linalg.solve(w_dP,w_P)

        #extract updated coefficients from p update/up matrix
        subset_coefficients = np.array([ [up[0,2]],
                                         [up[0,0]-1],
                                         [up[0,1]],
                                         [up[1,2]],
                                         [up[1,0]],
                                         [up[1,1]-1] ])

    if settings['DeformationModel'] == 'Quadratic':
        #order of SFP's P[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
        #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy
        A1 = 2*p[1] + p[1]**2 + + p[0]*p[3] 
        A2 = 2*p[0]*p[4] + 2*(1+p[1])*p[2]
        A3 = p[2]**2 + p[0]*p[5]
        A4 = 2*p[0]*(1+p[1])
        A5 = 2*p[0]*p[2]
        A6 = p[0]**2
        A7 = 0.5*(p[6]*p[3] + 2*(1+p[1])*p[7] + p[0]*p[9]) 
        A8 = p[2]*p[7] + p[1]*p[8] + p[6]*p[4] + p[0]*p[10] + p[8] + p[1]
        A9 = 0.5*(p[6]*p[5] + 2*(1+p[8])*p[2] + p[0]*p[11])
        A10 = p[6] + p[6]*p[1] + p[0]*p[7]
        A11 = p[0] + p[6]*p[2] + p[0]*p[8]
        A12 = p[0]*p[6]
        A13 = p[7]**2 + p[6]*p[9]
        A14 = 2*p[6]*p[10] + 2*p[7]*(1+p[8])
        A15 = 2*p[8]  + p[8]**2 + p[6]*p[11]
        A16 = 2*p[6]*p[7] 
        A17 = 2*p[6]*(1+p[8])
        A18 = p[6]**2

        #entries of w for update
        dA1 = 2*dp[1] + dp[1]**2 + + dp[0]*dp[3] 
        dA2 = 2*dp[0]*dp[4] + 2*(1+dp[1])*dp[2]
        dA3 = dp[2]**2 + dp[0]*dp[5]
        dA4 = 2*dp[0]*(1+dp[1])
        dA5 = 2*dp[0]*dp[2]
        dA6 = p[0]**2
        dA7 = 0.5*(dp[6]*dp[3] + 2*(1+dp[1])*dp[7] + dp[0]*dp[9]) 
        dA8 = dp[2]*dp[7] + dp[1]*dp[8] + dp[6]*dp[4] + dp[0]*dp[10] + dp[8] + dp[1]
        dA9 = 0.5*(dp[6]*dp[5] + 2*(1+dp[8])*dp[2] + dp[0]*dp[11])
        dA10 = dp[6] + dp[6]*dp[1] + dp[0]*dp[7]
        dA11 = dp[0] + dp[6]*dp[2] + dp[0]*dp[8]
        dA12 = dp[0]*dp[6]
        dA13 = dp[7]**2 + dp[6]*dp[9]
        dA14 = 2*dp[6]*dp[10] + 2*dp[7]*(1+dp[8])
        dA15 = 2*dp[8]  + dp[8]**2 + dp[6]*dp[11]
        dA16 = 2*dp[6]*dp[7] 
        dA17 = 2*dp[6]*(1+dp[8])
        dA18 = dp[6]**2

        #order of SFP's P[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
        #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy
        #w of current estimate of SFP's
        w_P = np.array([ [1+A1,       A2,       A3,       A4,     A5,      A6],
                         [ A7,       1+A8,      A9,      A10,    A11,     A12],
                         [A13,       A14,     1+ A15,    A16,    A17,     A18],
                         [0.5*p[3],  p[4],   0.5*p[5] , 1+p[1],  p[2],   p[0]],
                         [0.5*p[9],  p[10],  0.5*p[11],  p[7],  1+p[8],  p[6]],
                         [0,           0,       0,         0,     0,        1] 

                       ]).astype('double')
                        
        #w of current deltaP
        w_dP = np.array([ [1+dA1,       dA2,       dA3,       dA4,     dA5,      dA6],
                          [ dA7,       1+dA8,      dA9,      dA10,    dA11,     dA12],
                          [dA13,       dA14,     1+ dA15,    dA16,    dA17,     dA18],
                          [0.5*dp[3],  dp[4],   0.5*dp[5] , 1+dp[1],  dp[2],   dp[0]],
                          [0.5*dp[9],  dp[10],  0.5*dp[11],  dp[7],  1+dp[8],  dp[6]],
                          [0,           0,         0,          0,       0,         1]

                        ]).astype('double')
                        
        #P update matrix                
        up = np.linalg.solve(w_dP,w_P)
        subset_coefficients = np.array([ [up[3,5]],
                                         [up[3,3]-1],
                                         [up[3,4]],
                                         [2*up[3,0]],
                                         [up[3,1]],
                                         [2*up[3,2]],
                                         [up[4,5]],
                                         [up[4,3]],
                                         [up[4,4]-1],
                                         [2*up[4,0]],
                                         [up[4,1]],
                                         [2*up[4,2]] ])

    return subset_coefficients
#
def DeformationModelSubset(settings, p, xsi, eta):
    #p are the subset warp coefficients, abbreviated here as p for readability
    if settings['DeformationModel'] == 'Affine':
        #displace, stretch and shear subset in xy-coordinates (Affine):
        #order of SFP's p[j]: 0,  1,  2,  3,  4,  5
        #                     u   ux  uy  v   vx  vy
        xsi_d = (1+p[1])*xsi + p[2]*eta + p[0]
        eta_d = p[4]*xsi + (1+p[5])*eta + p[3]

    if settings['DeformationModel'] == 'Quadratic':
        #order of SFP's p[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
        #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy
        xsi_d = 0.5*p[3]*xsi**2 + p[4]*xsi*eta + 0.5*p[5]*eta**2 + (1+p[1])*xsi + p[2]*eta + p[0]
        eta_d = 0.5*p[9]*xsi**2 + p[10]*xsi*eta + 0.5*p[11]*eta**2 + p[7]*xsi + (1+p[8])*eta + p[6] 
    
    #xsi_eta_deformed = np.hstack([xsi_d, eta_d])
    return xsi_d, eta_d
#
def ReferenceSubset(F, delF, sub_size, xo, yo, settings):

    #extract  refrence subset intensity values, f, from mother image, F,
    #based on subset center coordinates
    if settings['InterpolateReferenceImage'] == 'Yes':

        yi = yo + settings['xsi']
        xi = xo + settings['eta']
        f = F(yi, xi)
        dfdy = delF[0](yi, xi)
        dfdx = delF[1](yi, xi)

    if settings['InterpolateReferenceImage'] == 'No':

        f = F[ yo - int(0.5*(sub_size-1)): yo + int(0.5*(sub_size-1)) + 1,
            xo - int(0.5*(sub_size-1)): xo + int(0.5*(sub_size-1)) + 1 ]
        # subset gradients
        # Fy = delF[0], Fx = delF[1]
        dfdy = delF[0][ yo - int(0.5*(sub_size-1)): yo + int(0.5*(sub_size-1)) + 1,
                        xo - int(0.5*(sub_size-1)): xo + int(0.5*(sub_size-1)) + 1 ]
        
        dfdx = delF[1][ yo - int(0.5*(sub_size-1)): yo + int(0.5*(sub_size-1)) + 1,
                        xo - int(0.5*(sub_size-1)): xo + int(0.5*(sub_size-1)) + 1 ]
        f = np.array([f.flatten(order = 'F')]).T

    #average subset intensity, and normalsied sum of squared differences
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    
    return f, f_mean, f_tilde, [dfdy, dfdx]
#
def DeformedSubset(G_interpolated, xsi_d, eta_d, xo, yo):

    #coordintes of intensity values of deformed subset/template
    #(all intesity values in the subset)
    yd = yo + eta_d
    xd = xo + xsi_d

    #extract  deformed subset intensity values, g, from mother image at sub-pixel coordinates
    g = np.array(G_interpolated(yd, xd))

    #determine average intensity value of subset g,
    # and normalised sum of squared differences of subset, g_tilde
    g_mean = g.mean()
    g_tilde = np.sqrt(np.sum((g[:]-g_mean)**2))

    return g, g_mean, g_tilde
#
def UpdateMeasurementPoints(settings, i_coeffs):
    
    if settings['DeformationModel'] == 'Affine':
        
        u, v = i_coeffs[0, :], i_coeffs[3, :]

    if settings['DeformationModel'] == 'Quadratic':
        
        u, v = i_coeffs[0, :], i_coeffs[6, :]
            
    #(introduce round-off error)
    #(if Yes introduce interpolation error)
    if settings['InterpolateReferenceImage'] == 'No':
        
        u = np.around(u).astype('int')
        v = np.around(v).astype('int')                

    settings['MeasurementPoints'] = settings['MeasurementPoints'] + np.vstack((
                                                                                u,
                                                                                v 
                                                                              )) 
    pass
#
def UpdateROI(settings):

    xo =  settings['MeasurementPoints'][0, :]
    yo = settings['MeasurementPoints'][1, :]
    sub_size = settings['SubsetSize']

    #ROI of new reference image
    x_min = np.min(xo) - 2*sub_size
    x_max = np.max(xo) + 2*sub_size

    y_min = np.min(yo) - 2*sub_size
    y_max = np.max(yo) + 2*sub_size

    ROI =  (x_min, y_min, x_max-x_min, y_max-y_min)
    
    return ROI
#
def ExitCriteriaLocal(settings, delta_p, hw):

    #hw is the half-width of the subset
    if settings['DeformationModel'] == 'Affine':

        hw_vector = np.array([[1,hw,hw,
                               1,hw,hw]])

    if settings['DeformationModel'] == 'Quadratic':

        hw_vector = np.array([1, hw, hw, 0.5*hw**2, hw**2, 0.5*hw**2,
                              1, hw, hw, 0.5*hw**2, hw**2, 0.5*hw**2])
    
    exit_criteria = np.sqrt(np.sum((delta_p*hw_vector)**2))

    return exit_criteria
#
def IncrementalLocal(settings, image_set, ROI):
        
    image0, image_target, increment, n_image_pairs = RefineImageSet(settings)

    #number of coefficients in deformation model
    if settings['DeformationModel'] == 'Affine':
        n_coeffs = 6
    else:
        n_coeffs = 12
    
    #storage array for deformation model coefficients of
    #EACH respective image pair
    relative_coeffs = np.zeros([(n_coeffs*n_image_pairs), 
                                 settings['MeasurementPoints'].shape[1]])
    
    #loop through all image pairs in the incremental range
    image_pair = 0
    for i_image in range(image0, image_target, increment):

        #coefficients at convergence for current (i'th) image pair
        i_coeffs = GaussNewtonLocal(settings, ROI, image_set, i_image, image_pair)

        #store coefficients at convergence of current image pair
        relative_coeffs[n_coeffs*image_pair:
                        n_coeffs*image_pair + n_coeffs, :] = i_coeffs

        #measurement point coordinates and ROI for next image pair
        UpdateMeasurementPoints(settings, i_coeffs)
        ROI = UpdateROI(settings)
        print('Image pair {} processed'.format(image_pair))

        image_pair += 1

    return relative_coeffs
#
def RefineImageSet(settings):

    #indices of datum and target images in image set
    image0 = settings['DatumImage']
    image_target = settings['TargetImage']
    #size of referencing increment in image set
    increment = settings['Increment']
    
    #number of image pairs to process for incremental referencing strategy
    n_image_pairs = int((image_target - image0)/increment)
    print('\nno. of image pairs:', n_image_pairs)

    return image0, image_target, increment, n_image_pairs
#   
def LocalMeasurementPoints(settings, ROI = np.zeros([4])):
    
    #default, case where the entire image should be populated with measurement points
    if np.sum(ROI) == 0:
        x_origin = 0
        y_origin = 0
        x_bound = settings['ImageColumns']
        y_bound = settings['ImageRows']

    #case where the ROI has been manually refined 
    else: 
        x_origin = ROI[0]
        y_origin = ROI[1]
        x_bound = ROI[0] + ROI[2]
        y_bound = ROI[1] + ROI[3]

    sub_size = int(settings['SubsetSize'])
    step_size = int(settings['StepSize'])

    #measurement point coordinates: defined in reference image, i.e subset centres
    yo, xo = np.meshgrid( np.arange( int(y_origin + 1*(sub_size-1) + step_size),
                                     int(y_bound - 1*(sub_size-1)),
                                     step_size), 

                          np.arange( int(x_origin + 1*(sub_size-1) + step_size),
                                     int(x_bound - 1*(sub_size-1)),
                                     step_size),     
                                                                
                          indexing = 'ij' )
                   
    #flatten measurement point coordinates to vectors
    xo = np.array([xo.flatten(order = 'F')]).T
    yo = np.array([yo.flatten(order = 'F')]).T
    measurement_points = np.vstack((xo.T, yo.T))

   
    return measurement_points
#
def FastInterpolation(image):

    #image dimensions
    ny = image.shape[0]
    nx = image.shape[1]

    #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    return image_interpolated
#
def GlobalMeasurementPoints(settings, ROI = np.zeros([4])):
    
    pass
#
def AbsoluteReferencing():

    pass
#
def GaussNewtonGlobal(settings, ROI, image_set, i_image):

    #return nodal_displacements
    pass
#
def EstimateDisplacementsFourier():
    
    pass
#
def IncrementalGlobal(settings, image_set, ROI):

    image0, image_target, increment, n_image_pairs = RefineImageSet(settings,
                                                                image_set, ROI)

    if settings['Discretisation'] == 'Global':

        for i_image in (image0, image_target + increment, increment):
            
            coefficients = GaussNewtonGlobal(settings, ROI, image_set, i_image)    

    #return dofs
    
    pass
#
def PlaneFitLS(Xo_camera):

    xo_c = Xo_camera[:, 0]
    yo_c = Xo_camera[:, 1]
    zo_c = Xo_camera[:, 2]
    
    #plane coefficient matrix
    A = np.array([[np.sum(xo_c*xo_c), np.sum(xo_c*yo_c), np.sum(xo_c)],
              
                  [np.sum(xo_c*yo_c), np.sum(yo_c*yo_c), np.sum(yo_c)],
              
                  [np.sum(xo_c), np.sum(yo_c), xo_c.shape[0]]])

    #plane fit RHS
    b = np.array([[-1*np.sum(xo_c*zo_c)],
                  [-1*np.sum(yo_c*zo_c)],
                  [-1*np.sum(zo_c)]])

    #coefficients
    C = np.linalg.solve(A, b)

    #Z coordinates from plane fit parameters and XY camera coordinates

    Z = -1*(C[0]*xo_c + C[1]*yo_c + C[2])

    return C, Z

def RotationMatrixFromBases(C, axis_world):

    #find the basis vectors of the best fit plane
    Zorigin = -1*(C[0][0]*axis_world[0, 0] + C[1][0]*axis_world[0, 1] + C[2][0])
    Zx = -1*(C[0][0]*axis_world[1, 0] + C[1][0]*axis_world[1, 1] + C[2][0])

    #three basis vectors of best fit plane
    g3 = np.array([C[0][0], C[1][0], 1])
    g1 = np.array([axis_world[1, 0] - axis_world[0, 0], axis_world[1, 1] - axis_world[0, 1], Zx - Zorigin])
    g2 = np.cross(g3, g1)

    #determine the rotation matrix fromm projections
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    e3 = np.array([0,0,1])
    E = np.vstack((e1, e2, e3))
    G = np.vstack((g1, g2, g3))
    R = np.zeros([3,3])
    
    for i in range(0,3):
        for j in range(0,3):

            R[i, j] = np.dot(E[i, :], G[j, :])/(norm(E[i, :])*norm(G[j, :]))

    return R
#
def ReadCalibrationParamters(settings, data):

    folder = settings['CalibrationParametersFolder']

    #left camera
    K0 = np.loadtxt('{}/K0.csv'.format(folder))
    R_0 = np.loadtxt('{}/R_0.csv'.format(folder))
    t_0 = np.loadtxt('{}/t_0.csv'.format(folder))
    r0 = np.loadtxt('{}/r0.csv'.format(folder))
    t_0 = np.array([t_0]).T

    #right camera
    K1 = np.loadtxt('{}/K1.csv'.format(folder))
    R_1 = np.loadtxt('{}/R_1.csv'.format(folder))
    t_1 = np.loadtxt('{}/t_1.csv'.format(folder))
    r1 = np.loadtxt('{}/r1.csv'.format(folder))
    t_1 = np.array([t_1]).T

    #homogeneous transformation matrics
    Rt_0 = np.vstack((np.hstack((R_0, t_0)), np.array([0,0,0,1])))
    Rt_1 = np.vstack((np.hstack((R_1, t_1)), np.array([0,0,0,1])))

    #camera projection matrices
    P0 = K0@Rt_0
    P1 = K1@Rt_1

    K0 = K0[0:3, 0:3]
    K1 = K1[0:3, 0:3]

    data['CalibrationParameters'] = [K0, K1, r0, r1, P0, P1] 

    return settings, data

def BestPlaneFit(settings, data, image_set_0, image_set_1, xo_local_L):

    #measurement point coordinates in reference image, camera coordinate system
    Xo_camera = data['Xo_camera']
    #measurement point coordinate in deformed imagem, coordinate system of camera
    Xd_camera = data['Xd_camera']
    #load calibration data
    K_L, K_R, r_L, r_R, P_L, P_R = data['CalibrationParameters']


    C, Z = PlaneFitLS(Xo_camera)

    #find local axis pixel coordinates in right image
    np.savetxt('measurement_points_Local.csv', xo_local_L, delimiter = ',')
    #temporarily change some settings
    settings_local = settings.copy()
    settings_local['Import'] = 'Yes'
    settings_local['MeasurementPointsName'] = 'measurement_points_Local'
    settings_local['CameraNo'] = 0
    p_specimen_L, p_specimen_R, settings_local, _ = StereoDICPlanar(settings_local, image_set_0, image_set_1, settings['ROI_L0'])
    xo_local_R = settings_local['MeasurementPoints_R0']

    #3D world coordinates on local axis coordinate system specimen surface
    axis_local = TriangulateOCV(P_L, P_R, UndistortPoints(xo_local_L.T, K_L, r_L).T,
                                          UndistortPoints(xo_local_R.T, K_R, r_R).T)
    
    #rotation matrix that brings the left camera coordinate system into the local CS on specimen surface
    R = RotationMatrixFromBases(C, axis_local)
    #translation from camera coordinate system to origin of locally defined axis CS
    t = np.array([axis_local[0, :]]).T
    R_t = np.vstack((np.hstack((R, t)), np.array([0,0,0,1])))

    #homogeneous 3D coordinates in camera coordinate system
    Xo_camera_ = np.ones([Xo_camera.shape[0], 4])
    Xo_camera_[:, 0:3] = Xo_camera
    Xd_camera_ = np.ones([Xd_camera.shape[0], 4])
    Xd_camera_[:, 0:3] = Xd_camera

    #transform coordinates from camera coordinate system to the locally defined CS
    Xo_w = np.linalg.inv(R_t)@Xo_camera_.T
    Xd_w = np.linalg.inv(R_t)@Xd_camera_.T
    U_w = Xd_w - Xo_w

    return Xo_w, Xd_w, U_w
#
def UndistortPoints(X, camera_matrix, dist_coeff):

    #number of coordinates to undistort
    n_points = X.shape[0]
    #reshape coordinates to appropriate shape for cv.undistortpoints function
    X_ = np.zeros((n_points,1,2), dtype=np.float32)
    X_[:, 0, :] = X[:, :]
    # do the actual undistort
    X_undistorted = cv.undistortPoints(X_,camera_matrix, dist_coeff, P=camera_matrix)

    return X_undistorted[:, 0, :]
#
def TriangulateOCV(Q0, Q1, x0, x1):

    coords_hom = cv.triangulatePoints(Q0, Q1, x0, x1)

    coords_3D = (coords_hom[:3, :]/coords_hom[3, :]).T

    return coords_3D
#
def VirtualStrainGauge(settings, xo, yo, U, V):

    #function can be used for either stereo or 2D-strain filtering

    size = settings['VSGFilterSize']

    n_subsets = xo.shape[0]
    kNN_VSG_indices = np.zeros([n_subsets, size**2])
    kNN_VSG_indices = np.zeros([n_subsets, size**2])
    kNN_VSG_indices = np.zeros([n_subsets, size**2])

    #storage array for each strain measurment
    ux = np.zeros_like(xo)
    uy = np.zeros_like(xo)
    vx = np.zeros_like(xo)
    vy = np.zeros_like(xo)

    #find the k nearest measurement points to each virtual strain gauge location
    for i in range(n_subsets):

        dist = np.sqrt((xo[i] - xo[:])**2 + (yo[i]- yo[:])**2)
        #indices of nearest subsests
        idx = np.argpartition(dist, size**2)
        #k minimum SSD between subset centre i and all keypoints
        kNN_VSG_indices[i, :] = idx[:size**2]

    #fit a polynomial to each measurment point using measurement data in its local neighborhood
    for i in range(n_subsets):

        #relative coordinates of measurement points around current point for strain filter
        x_rel = xo[kNN_VSG_indices[i, :].astype('int')] - xo[i]
        y_rel = yo[kNN_VSG_indices[i, :].astype('int')] - yo[i]

        #diplacements at measurement points around current point for strain filter
        u = U[kNN_VSG_indices[i, :].astype('int')]
        v = V[kNN_VSG_indices[i, :].astype('int')]

        if settings['VSGFilterOrder'] == 'Linear':
            #strain filter coefficient matrix
            A = np.ones([size**2, 3])
            A[:, 1] = x_rel
            A[:, 2] = y_rel

        if settings['VSGFilterOrder'] == 'Quadratic':
            #strain filter coefficient matrix
            A = np.ones([size**2, 6])
            A[:, 1] = x_rel
            A[:, 2] = y_rel
            A[:, 3] = x_rel**2
            A[:, 4] = x_rel*y_rel
            A[:, 5] = y_rel**2

        #LS solution for the strain filter
        u_coeffs = np.linalg.solve(A.T@A, A.T@u)
        v_coeffs = np.linalg.solve(A.T@A, A.T@v)

        #return coefficients for displacement gradients
        ux[i] = u_coeffs[1]
        uy[i] = u_coeffs[2]

        vx[i] = v_coeffs[1]
        vy[i] = v_coeffs[2]

    #green-lagrange strain tensor components
    Ex =  ux + 0.5*ux*ux + 0.5*vx*vx
    Ey =  vy + 0.5*vy*vy + 0.5*uy*uy
    Exy = 0.5*(uy + vx + ux*uy + vx*vy)    

    return Ex, Ey, Exy
#
def DepthFromTwoViewsCameraCS(settings, data):

    #find the depth measurements in the reference and deformed positions
    #in the coordinate system of the reference(left) camera
 
    #measurement points/subset centre coordinates
    xo_L = settings['MeasurementPoints_L0']
    xo_R = settings['MeasurementPoints_R0']

    #total displacements, both image series
    U_L = data['U_L']
    U_R = data['U_R']

    #load the internal and stereo calibration parameters
    K_L, K_R, r_L, r_R, P_L, P_R = data['CalibrationParameters']

    #displaced subset centre positions in deformed image
    xd_L = xo_L + U_L
    xd_R = xo_R + U_R

    #3D world coordinates in (left) camera coordinate system, reference position
    Xo_camera = TriangulateOCV(P_L, P_R, UndistortPoints(xo_L.T, K_L, r_L).T, UndistortPoints(xo_R.T, K_R, r_R).T)
    #3D world coordinates in (left) camera coordinate system, after deformation
    Xd_camera = TriangulateOCV(P_L, P_R, UndistortPoints(xd_L.T, K_L, r_L).T, UndistortPoints(xd_R.T, K_R, r_R).T)
    #3D displacements in (left) camera coordinate system
    U_camera = Xd_camera - Xo_camera

    #save data
    data['Xo_camera'] = Xo_camera
    data['Xd_camera'] = Xd_camera
    data['U_camera'] = U_camera 

    #u = U_camera[:, 0]

    return settings, data












































#------------------------------------------------------------------------------------
# def GaussQuadrature2D(order):

#     xsi_eta, w2 = gq.ExportGQrundic(order)

#     return xsi_eta, w2
# #    
# def PlanarDICGlobal(settings, image_set, ROI):
#     #measurement point coordinates defined in datum image
#     #xsi, eta: local coordinates defined within subset(s)
#     if settings['CameraNo'] in {0, None}:

#         #load measurement point from external file
#         if settings['ImportMeasurementPointsFromExternal'] == 'Yes':
            
#             #mesh parser functions come here
#             # measurement_points = open('measurement_points.csv', 'rb')
#             # measurement_points = loadtxt(measurement_points, delimiter = ",")
#             # measurement_points = np.array(measurement_points)

#             pass

#         #should also have an option to create the mesh based on settings and
#         #measurement point coordinates
        
#         #define measurement points using the settings specified in the config file 
#         if settings['ImportMeasurementPointsFromExternal'] == 'No':
            
#             mesh = RectangularMeshQuadrilaterals(settings, image_set, ROI)
#             #append the measurement points to the settings file
#             #store measurement points for later triangulation of displacement coefficients
#             #np.savetxt("measurement_points.csv", measurement_points, delimiter = ",")
        
#         #save measurement point coordinates in datum image of left camera as reference
#         #for later use in cross correlation
#         if settings['MeasurementType'] in {'StereoCurved', 'StereoPlanar'}:
#             settings['MeasurementPoints_L0'] = settings['MeasurementPoints']
#             settings['ROI_L0'] = ROI

#     if settings['CorrelationReferenceStrategy'] == 'Incremental':
#         coefficients = IncrementalLocal(settings, image_set, ROI)

#     elif settings['CorrelationReferenceStrategy'] == 'Absolute':
#         coefficients = AbsoluteReferencing(settings, image_set, ROI)

#     return coefficients
# #
# def RectangularMeshQuadrilaterals(settings, ROI):

#     element_size = settings['ElementSize']
#     xsi_eta, w2 = GaussQuadrature2D(settings['GQOrder'])

#     #default, case where the entire image should be populated with nodes
#     if np.sum(ROI) == 0:
#         x_origin = 0
#         y_origin = 0
#         x_bound = settings['ImageColumns']
#         y_bound = settings['ImageRows']

#     #case where the ROI has been manually refined 
#     else: 
#         x_origin = ROI[0]
#         y_origin = ROI[1]
#         x_bound = ROI[0] + ROI[2]
#         y_bound = ROI[1] + ROI[3]

#     if settings['ElementType'] == 'Q4':

#         pass

#     if settings['ElementType'] == 'Q8':

#         pass

#     if settings['ElementType'] == 'Q9':

#         #structured grid of x and y coordinates
#         yn, xn = np.meshgrid( np.arange(y_origin,
#                                         y_bound,
#                                         (element_size-1)/2),

#                               np.arange(x_origin,
#                                         x_bound,
#                                         (element_size-1)/2),

#                               indexing = 'ij')

#         #number of nodes in the x and y directions
#         n_nodes_y = yn.shape[0]
#         n_nodes_x = xn.shape[1]
#         n_elements = int(((n_nodes_x-1)/2)*((n_nodes_y-1)/2))

#         #node XY coordinates as vectors
#         node_coords = np.vstack(( xn.ravel(order = 'F'),
#                                   yn.ravel(order = 'F') )).T
#         n_nodes = node_coords.shape[0]

#         #create the Q9 mesh and label the nodes
#         #(should add function here that exports a graphic of the mesh showing
#         #the node numbering)
#         mesh_node_no = np.arange(0, n_nodes).reshape(n_nodes_y, n_nodes_x, order = 'F')
        
#         #element connectivity table
#         element_conn = np.zeros([n_elements, 9]).astype(int)
#         l = 0
#         for j in range(0, int(((n_nodes_x-1)/2)) + 2, 2):
#             #rows
#             for i in range(0, int(((n_nodes_x-1)/2)) + 2, 2):

#                 element_conn[l, :] = np.array([ mesh_node_no[i, j], mesh_node_no[i+2, j], mesh_node_no[i+2, j+2], mesh_node_no[i, j+2], 
#                                                 mesh_node_no[i+1, j], mesh_node_no[i+2, j+1], mesh_node_no[i+1, j+2], mesh_node_no[i, j+1], mesh_node_no[i+1, j+1] ])
#                 l = l + 1

#         #number the displacement degrees of freedom
#         dof = np.vstack(( np.array(np.arange(n_nodes)),
#                           np.array(n_nodes + np.arange(n_nodes)) )).T

#         #total number if degrees of freedom
#         n_dof = 2*n_nodes

#         #node_coords, element_conn, dof, N_dof, n_elements

#     mesh = dict()
#     mesh['NodeCoordinates'] = node_coords
#     mesh['ElementConnectivity'] = element_conn
#     mesh['DOFIndices'] = dof
#     mesh['n_Nodes'] = n_nodes
#     mesh['n_Elements'] = n_elements
#     mesh['n_IP'] = n_IP
#     #mesh['n_Nodes']
    
#     return mesh
