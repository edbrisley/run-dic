"""Subset-based 2D Digital Image Correlation library

    Functions in this library are used to perform Subset-based 2D-DIC.

    Notes
    -----
    This is an open-source library.

    This code was setup by Ed Brisley, in collaboratin with Prof Gerhard Venter and
    the Materials Optimisation and Design (MOD) research group at Stellenbosch University.
    June 30, 2023
    
    References
    ----------
    [1] github repository link: 
"""
#-------------------------------------------------------------------------------------
# Importing libraries
#-------------------------------------------------------------------------------------
from tokenize import Double
import numpy as np
import numpy.linalg as la
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

 
#-------------------------------------------------------------------------------------
# Versioning the code
# 0.1.0 - 
# Version 0.2.0 - 
# Version 0.2.1 - 
#-------------------------------------------------------------------------------------
__version__="0.1.0"


#-------------------------------------------------------------------------------------
def LoadSettings(image_set):
    """Load DIC settings specified by user
    
    Load DIC settings specified by user in the file, Settings.ini, they are:

    [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
     sub_frequency, corr_refstrat, calibrate]

    Store the specified settings in an array, settings, to be used throughout the rest
    of the program. Extract the dimensions of the images contained in the image set and
    add them to the settings array.
    
    Parameters
    ----------
        image_set    : List of names of images in the image set
        
    Returns
    -------
        settings    : [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                       sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    """

    ##load the configuration file containing the DIC settings
    configur = ConfigParser()
    configur.read('Settings.ini')
    sub_size = configur.getint('Subsets', 'SubsetSize')
    sub_frequency = configur.getint('Subsets', 'SubsetFrequency')
    GF_stddev = configur.getfloat('Filters', 'GaussianFilterStdDev')
    GF_filtsize = configur.getint('Filters', 'GaussianFilterSize')
    SF_order = configur.getint('Miscellaneous', 'ShapeFunctionOrder')
    corr_refstrat = configur.get('Miscellaneous', 'CorrelationReferenceStrategy')
    calibrate = configur.get('Miscellaneous', 'Calibration')
    sub_shape = configur.get('Subsets', 'SubsetShape')

    #store setup parameters
    settings = dict()
    settings['SubsetSize'] = sub_size
    settings['ShapeFunctionOrder'] = SF_order
    settings['GaussianFilterStdDev'] = GF_stddev
    settings['GaussianFilterSize'] = GF_filtsize
    settings['CorrelationReferenceStrategy'] = corr_refstrat
    settings['SubsetFrequency'] = sub_frequency

    #determine image dimensions
    img_0 = cv.imread("images2/{}".format(image_set[0]),0)
    img_rows = img_0.shape[0]
    img_columns = img_0.shape[1]
    settings['ImageColumns'] = img_columns
    settings['ImageRows'] = img_rows
    settings['Calibration'] = calibrate
    settings['SubsetShape'] = sub_shape
    
    return settings


#-------------------------------------------------------------------------------------
def LoadImages():
    """Load images from directory on which to perform DIC
    
    Specify the directory where the images are stored.
    Read the image names from the directory and store them in a list: image_set.

    Parameters
    ----------
        none
        
    Returns
    -------
        image_set    : List containing image names in the working directory,
                       in the order they are read
    """

    #define image directory
    current_working_directory = os.getcwd()
    #detect OS 
    if platform.system() == 'Linux':
        image_folder = '/images2'
    #windows machine (need to add macOS option)
    else:
        image_folder = '\images2'
    #location of images for DIC measurement
    image_location = current_working_directory + image_folder
    #read images in directory
    image_set = []
    for filename in os.listdir(image_location):    
        image_set.append(filename)
    
    return image_set


#-------------------------------------------------------------------------------------
class DICImage:
    """General DIC image class

    Assign settings variables associated with DIC to image object.
    Inherited by the ReferenceImage and DeformedImage classes
    
    Parameters
    ----------
        image           : Grayscale (Intensity) image; range: [0, 255]
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    Attributes
    -------
        image           : Normalised image; range: [0, 1]
        image_8bit      : 8 bit depth image; range: [0, 255]
                          reserved to allow for ROI selection and feature detection
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]

    """

    def __init__(self, image, settings):
        
        #keep intensity image in range [0,255] for keypoint matching
        self.image_8bit = image
        #normalized intensity image
        self.image = cv.normalize(image.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
        self.sub_size =  int(settings['SubsetSize'])
        self.SF_order = settings['ShapeFunctionOrder']
        self.GF_stddev = settings['GaussianFilterStdDev']
        self.GF_filtsize = int(settings['GaussianFilterSize'])
        self.corr_refstrat = settings['CorrelationReferenceStrategy']
        self.frequency = int(settings['SubsetFrequency'])
        self.img_rows = settings['ImageRows']
        self.img_columns = settings['ImageColumns']
        self.sub_halfwidth = (self.sub_size-1)/2


#-------------------------------------------------------------------------------------       
class ReferenceImage(DICImage):
    """Reference image class for DIC

    Reference image for measuring deformation during subset-based 2D-DIC.
    Inherits all attributes from the DICImage class' __init__ method.
    
    Parameters
    (Inherited from DICImage class)
    ----------
        image           : Normalised image; range: [0, 1]
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    
    Attributes
    (Inherited from DICImage class)
    -------
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
        image_8bit      : 8 bit depth image; range: [0, 255]
                          reserved to allow for ROI selection and feature detection
    
    Attributes
    (__init__ method)
    -------
        image           : Normalised image after applying a Gaussian blur;
                          range: [0, 1]
        F_gradX         : Gradient of image intensity in X-direction
        F_gradY         : Gradient of image intensity in Y-direction

    Attributes
    (CreateSubsets method)
    -------
        sub_centers     : XY coordinates of all subsets' centers in the mother image
        x, y            : Relative xy coordinates within subset, specified by sub_size
                          These relative coordinates are the same within all subsets
        P               : Affine transformation coefficients
    """

    def __init__(self,image,settings):
        
        #inherit settings and intensity image from DICImage Class
        super().__init__(image,settings)

        #blur reference image with Gaussian filter
        self.image = cv.GaussianBlur(self.image,
                                    (self.GF_filtsize,self.GF_filtsize),
                                    self.GF_stddev)
                                    
        #gradient of F (reference image) in xy directions
        grad = np.array(np.gradient(self.image),dtype= float)
        self.F_gradY = grad[0]
        self.F_gradX = grad[1]

        #-------------------------------------------------------------------------------------
    def CreateSubsets(self, settings, ROI_coords = np.zeros([4])):
        """Create image subsets
    
    Store XY coordinates of each subset center
    (XY coordinates are the global coordinates in the mother image)
    
    Store xy relative coordinates of pixels within subset, these coordinates are the same
    for all subsets
    (xy coordinates are the relative local coordinates within the subset)

    Parameters
    ----------
        settings        : [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                          sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
        ROI_coords      : Coordinates that define the region of interest for measurements in
                          the reference image. Default ROI is the entire reference image
    
    Returns
    -------
        x, y            : Relative xy coordinates within subset, specified by sub_size
                          These relative coordinates are the same within all subsets
        sub_centers     : XY coordinates of all subsets' centers
        P               : Shape function parameters
    """
        #create coordinates of pixels in subset relative to center, x, y
        #(constant subset size-these relative coordinates are the same for all subsets)
        #create coordinates for subset centers
        #self.x, self.y, self.sub_centers = CreateSubsets(settings)

        #default, condition where the entire image should be populated with subsets
        if np.sum(ROI_coords) == 0:
            x_origin = 0
            y_origin = 0
            x_bound = settings['ImageColumns']
            y_bound = settings['ImageRows']

        #case where the ROI has been manually refined 
        else: 
            x_origin = ROI_coords[0]
            y_origin = ROI_coords[1]
            x_bound = ROI_coords[0] + ROI_coords[2]
            y_bound = ROI_coords[1] + ROI_coords[3]

            #fetch setup variables
        sub_size = int(settings['SubsetSize'])
        sub_freq = int(settings['SubsetFrequency'])

        #create subset centers XY coordinates as MESHGRIDS
        sub_centers_y, sub_centers_x = np.meshgrid(
                                       np.arange(
                                            int(y_origin + 5*(sub_size-1) + sub_freq),
                                            int(y_bound - 5*(sub_size-1)),
                                            sub_freq),
                                       np.arange(
                                            int(x_origin + 5*(sub_size-1) + sub_freq),
                                            int(x_bound - 5*(sub_size-1)),
                                            sub_freq),
                                            indexing = 'ij')

        #flatten subset centers XY coordinates to vectors
        sub_centers_x = np.array([sub_centers_x.flatten(order = 'F')]).T
        sub_centers_y = np.array([sub_centers_y.flatten(order = 'F')]).T
        self.sub_centers = np.vstack((sub_centers_x.T,sub_centers_y.T))

        #create subset xy relative coordinates
        [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                            np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                            indexing='ij')

        #flatten subset xy relative coordinates to vectors
        self.x = np.array([x.flatten(order = 'F')]).T
        self.y = np.array([y.flatten(order = 'F')]).T

        #shape function parameters of F, dependent on SF order
        if self.SF_order == 0:
            self.P = np.zeros([4, self.sub_centers.shape[1]])
        elif self.SF_order == 1:
            self.P = np.zeros([6, self.sub_centers.shape[1]]) 
        else:
            self.P = np.zeros([12, self.sub_centers.shape[1]])

    #-------------------------------------------------------------------------------------
    def CreateSubsetsSample14(self, settings, ROI_coords = np.zeros([4])):

            #fetch setup variables
        sub_size = int(settings['SubsetSize'])
        sub_freq = int(settings['SubsetFrequency'])

        #create subset centers XY coordinates as MESHGRIDS
        #create subset centers XY coordinates as MESHGRIDS
        sub_centers_y, sub_centers_x = np.meshgrid(np.arange(25,
                                                            570,
                                                            sub_freq),
                                                    np.arange(25,
                                                            2030,
                                                            sub_freq),
                                                    indexing = 'ij')

        #flatten subset centers XY coordinates to vectors
        sub_centers_x = np.array([sub_centers_x.flatten(order = 'F')]).T
        sub_centers_y = np.array([sub_centers_y.flatten(order = 'F')]).T
        self.sub_centers = np.vstack((sub_centers_x.T,sub_centers_y.T))

        #create subset xy relative coordinates
        [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                            np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                            indexing='ij')

        #flatten subset xy relative coordinates to vectors
        self.x = np.array([x.flatten(order = 'F')]).T
        self.y = np.array([y.flatten(order = 'F')]).T

        #shape function parameters of F, dependent on SF order
        if self.SF_order == 0:
            self.P = np.zeros([4, self.sub_centers.shape[1]])
        elif self.SF_order == 1:
            self.P = np.zeros([6, self.sub_centers.shape[1]]) 
        else:
            self.P = np.zeros([12, self.sub_centers.shape[1]])

#-------------------------------------------------------------------------------------
class DeformedImage(DICImage):
    """Deformed image class for DIC

    Deformed image in measuring deformation during subset-based 2D-DIC.
    Inherits all attributes from the DICImage class' __init__ method.
    
    Parameters
    (Inherited from DICImage class)
    ----------
        image           : Normalised image; range: [0, 1]
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    
    Attributes
    (Inherited from DICImage class)
    -------
        settings        : DIC settings array as specified by user, 
                          [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                           sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
        image_8bit      : 8 bit depth image; range: [0, 255]
                          reserved to allow for ROI selection and feature detection

    Attributes
    (__init__ method)
    -------
        image           : Normalised image after applying a Gaussian blur;
                          range: [0, 1]
        G_interpolated  : Interpolated image; range: [0, 1]
                          (intensity values range changed from discrete XY
                           integer-pixel locations to a continuos XY domain)


    Attributes
    (InitCorrelationParams)
    -------
        sub_centers     : XY coordinates of all subsets' centers in the mother image
        x, y            : Relative xy coordinates within subset, specified by sub_size
                          These relative coordinates are the same within all subsets
        P               : Affine transformation coefficients
        corr_coeff      : Correlation coefficients at convergence (all subsets)
        stop_val        : Exit criteria at convergence (all subsets)
        iterations      : No. of iterations at convergence (all subsets)

    """

    def __init__(self,image,settings):

        #inherit settings and intensity image from DICImage Class
        super().__init__(image,settings)

        #blur reference image with Gaussian filter 
        self.image = cv.GaussianBlur(self.image,
                                    (self.GF_filtsize,self.GF_filtsize),
                                    self.GF_stddev)
        #interpolation of deformed image
        self.G_interpolated = FastInterpolation(self.image)

    #auxiliary method
    def InitCorrelationParams(self, F):
        """Initialise correletaion parameters for deformed image object

            Initial estimates for subset centre coordinates and SFPs are set as those of the
            reference image.
 
            Parameters
            ----------
                F       : Reference image object
            Returns
            -------
                sub_centers     : XY coordinates of all subsets' centers in the mother image
                x, y            : Relative xy coordinates within subset, specified by sub_size
                                  These relative coordinates are the same within all subsets
                P               : Affine transformation coefficients
                corr_coeff      : Correlation coefficients at convergence (all subsets)
                stop_val        : Exit criteria at convergence (all subsets)
                iterations      : No. of iterations at convergence (all subsets)
        """
        #initialise subset center positions as those of the reference image
        #initialise SFPs vectors
        self.sub_centers, self.P, self.x, self.y = F.sub_centers, F.P, F.x, F.y
        #variables to save correlation run results at convergence
        self.corr_coeff = np.zeros([1,self.sub_centers.shape[1]])
        self.stop_val = np.zeros([1,self.sub_centers.shape[1]])
        self.iterations = np.zeros([1,self.sub_centers.shape[1]])


#-------------------------------------------------------------------------------------
def RefSubsetInfo(F, i):
    """Extract reference subset, f, from reference image, F

    Extract subset f from F, based on the subset center coordinates of f - specified by
    index, i

    Determine average intensity value of all pixels in subset f - f_mean

    Determine normalised sum of squared differences of subset - f_tilde

    Determine subset spatial gradients dfdx and dfdy from F.Fgrad - specified by index, i 


    Parameters
    ----------
        F       : Reference image object
        i       : Index specifying the XY coordinates of the reference subset center
    Returns
    -------
        f       : Reference subset intensity values
        f_mean  : Average subset intensity value
        f_tilde : Normalised sum of squared differences of subset
        dfdx    : Subset intensity gradient in x-direction
        dfdy    : Subset intensity gradient in y-direction
    """

    #fetch XY coordinates of i'th subset center
    centerx, centery = np.array([F.sub_centers[0][i],F.sub_centers[1][i]])

    #extract  refrence subset intensity values, f, from mother image, F,
    #based on subset center coordinates
    f = F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]

    #extract subset spatial gradients
    dfdy = F.F_gradY[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
    dfdx = F.F_gradX[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]

    #average subset intensity, and normalsied sum of squared differences
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    f = np.array([f.flatten(order = 'F')]).T

    return f, f_mean, f_tilde, dfdx, dfdy


#-------------------------------------------------------------------------------------
def Hessian(dfdx, dfdy, F):
    """Determine the Hessian of the reference subset

    The Hessian matrix of the reference subset stays constant during the optimization
    iterations

    Determine procuct of subset gradient [dfdx, dfdy] and warp function gradient dWdP ??

    Determine Hessian matrix as fgrad_X_dWdP.T*fgrad_X_dWdP.T

    Parameters
    ----------
        dfdx          : Subset spatial gradient of intensity values in x
        dfdy          : Subset spatial gradient of intensity values in y
        F             : Reference image object

    Returns
    -------
        Hess          : Reference subset Hessian matrix
        fgrad_X_dWdP  : Procuct of subset gradient [dfdx, dfdy] and 
                        warp function gradient dWdP
    """

    #flatten subset intensity gradients to vectors
    dfdx = np.array([dfdx.flatten(order = 'F')]).T
    dfdy = np.array([dfdy.flatten(order = 'F')]).T

    #fetch subset relative coordinates
    x = F.x
    y = F.y

    #Affine transformation
    if F.P.shape[0] == 6:
        #procuct of subset gradient [dfdx, dfdy] and warp function gradient dWdP 
        fgrad_X_dWdP = np.array([
                                dfdx[:,0]*1,
                                dfdx[:,0]*x[:,0],
                                dfdx[:,0]*y[:,0],
                                dfdy[:,0]*1,
                                dfdy[:,0]*x[:,0],
                                dfdy[:,0]*y[:,0]
                                ]).T

    #NL transformation
    if F.P.shape[0] == 12:
        #procuct of subset gradient [dfdx, dfdy] and warp function gradient dWdP
        fgrad_X_dWdP = np.array([
                                dfdx[:,0]*1,
                                dfdx[:,0]*x[:,0],
                                dfdx[:,0]*y[:,0],
                                dfdx[:,0]*0.5*x[:,0]**2,
                                dfdx[:,0]*x[:,0]*y[:,0],
                                dfdx[:,0]*0.5*y[:,0]**2,
                                dfdy[:,0]*1,
                                dfdy[:,0]*x[:,0],
                                dfdy[:,0]*y[:,0],
                                dfdy[:,0]*0.5*x[:, 0]**2,
                                dfdy[:,0]*x[:,0]*y[:,0],
                                dfdy[:,0]*0.5*y[:,0]**2
                                ]).T


    #compute hessian
    Hess = np.dot(fgrad_X_dWdP.T, fgrad_X_dWdP)   
    return Hess, fgrad_X_dWdP


##-------------------------------------------------------------------------------------
def AffineTrans(G, i):
    """Deform the subset using an affine transformation

    Deform the subset using affine transformation coefficients/ shape function
    parameters (SFP's) of the current iteration.

    Parameters
    ----------
        G               : Deformed image object
        i               : Index specifying the XY coordinates of the deformed
                          subset center

    Returns
    -------
        deformed_subset : Deformation of subset at all xy (relative subset) coordinates
    """


    #fetch affine transformation coefficients/shape function parameters (SFP's)
    P = G.P[:,i]

    #fetch subset relative coordinates
    x = G.x
    y = G.y

    #displace, stretch and shear subset in xy-coordinates (14):
    #order of SFP's P[j]: 0,  1,  2,  3,  4,  5
    #                     u   ux  uy  v   vx  vy
    deformed_x = (1+P[1])*x + P[2]*y + P[0]
    deformed_y = P[4]*x + (1+P[5])*y + P[3]
    deformed_subset = np.hstack([deformed_x, deformed_y])

    return deformed_subset


#-------------------------------------------------------------------------------------
def DefSubsetInfo(F, G, deformed_subset, i):
    """Extract deformed subset, g, from deformed image, G

    Extract subset g from G, based on the subset center coordinates of g - specified 
    by the index, i; and the deformation of the subset determined using the function
    AffineTransform

    Determine average intensity value of all pixels in subset g - g_mean

    Determine normalised sum of squared differences of subset g - g_tilde

    Parameters
    ----------
        G               : Deformed image object
        deformed_subset : Deformation of subset in xy coordinates
        i               : Index specifying the XY coordinates of the deformed
                          subset center
    Returns
    -------
        g               : Deformed subset intensity values
        g_mean          : Average intensity value of deformed subset
        g_tilde         : Normalised sum of squared differences of deformed subset
    """

    #fetch number of intensity values in subset
    N_points = deformed_subset.shape[0]

    #fetch XY coordinates of i'th subset center in reference coordinate frame
    centerx, centery = np.array([F.sub_centers[0][i],F.sub_centers[1][i]])

    #add deformed xy coordinates (determined using AffineTransform)
    #to the XY coordinates of the subset center
    #this gives the XY coordinates of the deformed subset in the mother image
    Y = centery*np.ones(N_points) + deformed_subset[:,1]
    X = centerx*np.ones(N_points) + deformed_subset[:,0]

    #extract  deformed subset intensity values, g, from mother image, G.Interpolated,
    #based on deformed subset XY coordinates
    g = np.array([G.G_interpolated(Y, X)]).T
    # for m in range(0,N_points):
    #     g[m] = G.G_interpolated(Y[m],X[m])

    #determine average intensity value of subset g,
    # and normalised sum of squared differences of subset, g_tilde
    g_mean = g.mean()
    g_tilde = np.sqrt(np.sum((g[:]-g_mean)**2))

    return g, g_mean, g_tilde


#-------------------------------------------------------------------------------------
def UpdateSFP(P, dP):
    """Update SFP's/Affine transformation coefficients based on SFP's (P) of current
       iteration and deltaP,  the inverted iterative improvement of the SFP's.

    a Matrix w(P) is populated with the current estimate of P. a Matrix w(dP) is
    populated with the terms in dP (inverted iterative improvement of P). The updated
    SFP's (Pupdate) are determined by extracting terms from the product of w(P)*w(dP)^-1.

    Parameters
    ----------
        P       : SFP's vector
        dP      : Inverted iterative improvement of SFP's

    Returns
    -------
        Pupdate : Updated estimate of SFP's
    """
    


    #Affine transformation
    if P.shape[0] == 6:
        #w of current estimate of SFPs
        #order of SFP's P[1]: 0,  1,  2,  3,  4,  5
        #                     u   ux  uy  v   vx  vy
        w_P = np.array([
                        [1+P[1],    P[2],    P[0]],
                        [ P[4],    1+P[5],   P[3]],
                        [  0,        0,        1 ] 
                        ]).astype('double')

        #w of current deltaP               
        w_dP = np.array([
                        [1+dP[1],     dP[2],    dP[0]],
                        [ dP[4],     1+dP[5],   dP[3]],
                        [   0,          0,        1  ]
                        ]).astype('double')

        #P update matrix                
        up = np.linalg.solve(w_dP,w_P)

        #extract updated SFP's from P update/up matrix
        Pupdate = np.array([
                            [up[0,2]],
                            [up[0,0]-1],
                            [up[0,1]],
                            [up[1,2]],
                            [up[1,0]],
                            [up[1,1]-1]
                        ])
    #NL transformation
    if P.shape[0] == 12:
        #order of SFP's P[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
        #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy
        A1 = 2*P[1] + P[1]**2 + + P[0]*P[3] 
        A2 = 2*P[0]*P[4] + 2*(1+P[1])*P[2]
        A3 = P[2]**2 + P[0]*P[5]
        A4 = 2*P[0]*(1+P[1])
        A5 = 2*P[0]*P[2]
        A6 = P[0]**2
        A7 = 0.5*(P[6]*P[3] + 2*(1+P[1])*P[7] + P[0]*P[9]) 
        A8 = P[2]*P[7] + P[1]*P[8] + P[6]*P[4] + P[0]*P[10] + P[8] + P[1]
        A9 = 0.5*(P[6]*P[5] + 2*(1+P[8])*P[2] + P[0]*P[11])
        A10 = P[6] + P[6]*P[1] + P[0]*P[7]
        A11 = P[0] + P[6]*P[2] + P[0]*P[8]
        A12 = P[0]*P[6]
        A13 = P[7]**2 + P[6]*P[9]
        A14 = 2*P[6]*P[10] + 2*P[7]*(1+P[8])
        A15 = 2*P[8]  + P[8]**2 + P[6]*P[11]
        A16 = 2*P[6]*P[7] 
        A17 = 2*P[6]*(1+P[8])
        A18 = P[6]**2

        #entries of w for update
        dA1 = 2*dP[1] + dP[1]**2 + + dP[0]*dP[3] 
        dA2 = 2*dP[0]*dP[4] + 2*(1+dP[1])*dP[2]
        dA3 = dP[2]**2 + dP[0]*dP[5]
        dA4 = 2*dP[0]*(1+dP[1])
        dA5 = 2*dP[0]*dP[2]
        dA6 = P[0]**2
        dA7 = 0.5*(dP[6]*dP[3] + 2*(1+dP[1])*dP[7] + dP[0]*dP[9]) 
        dA8 = dP[2]*dP[7] + dP[1]*dP[8] + dP[6]*dP[4] + dP[0]*dP[10] + dP[8] + dP[1]
        dA9 = 0.5*(dP[6]*dP[5] + 2*(1+dP[8])*dP[2] + dP[0]*dP[11])
        dA10 = dP[6] + dP[6]*dP[1] + dP[0]*dP[7]
        dA11 = dP[0] + dP[6]*dP[2] + dP[0]*dP[8]
        dA12 = dP[0]*dP[6]
        dA13 = dP[7]**2 + dP[6]*dP[9]
        dA14 = 2*dP[6]*dP[10] + 2*dP[7]*(1+dP[8])
        dA15 = 2*dP[8]  + dP[8]**2 + dP[6]*dP[11]
        dA16 = 2*dP[6]*dP[7] 
        dA17 = 2*dP[6]*(1+dP[8])
        dA18 = dP[6]**2

        #order of SFP's P[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
        #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy

        #w of current estimate of SFP's
        w_P = np.array([
                        [1+A1,       A2,       A3,       A4,     A5,      A6],
                        [ A7,       1+A8,      A9,      A10,    A11,     A12],
                        [A13,       A14,     1+ A15,    A16,    A17,     A18],
                        [0.5*P[3],  P[4],   0.5*P[5] , 1+P[1],  P[2],   P[0]],
                        [0.5*P[9],  P[10],  0.5*P[11],  P[7],  1+P[8],  P[6]],
                        [0,           0,       0,         0,     0,        1]
                        ]).astype('double')

        #w of current deltaP
        w_dP = np.array([
                        [1+dA1,       dA2,       dA3,       dA4,     dA5,      dA6],
                        [ dA7,       1+dA8,      dA9,      dA10,    dA11,     dA12],
                        [dA13,       dA14,     1+ dA15,    dA16,    dA17,     dA18],
                        [0.5*dP[3],  dP[4],   0.5*dP[5] , 1+dP[1],  dP[2],   dP[0]],
                        [0.5*dP[9],  dP[10],  0.5*dP[11],  dP[7],  1+dP[8],  dP[6]],
                        [0,           0,         0,          0,       0,         1]
                        ]).astype('double')

        #P update matrix                
        up = np.linalg.solve(w_dP,w_P)
        Pupdate = np.array([
                            [up[3,5]],
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
                            [2*up[4,2]],
                           ])


    return Pupdate

#
#-------------------------------------------------------------------------------------
def StopCriteria(dP, zeta):
    """Determine the value of the convergence parameter for the current estimate
       of the SFP's.

    The convergence parameter is computed as the normalised sum of squares of the
    product of P and [1,zeta,zeta,1,zeta,zeta]

    Parameters
    ----------
        dP                     : Inverted iterative improvement
                                 for current iteration of SFP's
        zeta                   : Halfwidth of the square subset

    Returns
    -------
        convergence_parameter  : Convergence parameter for current iteration of SFP's
    """

    #create zeta vector and compute convergence_parameter
    #affine transformation
    if dP.shape[0] == 6:
        zeta_vector = np.array([[1,zeta,zeta,1,zeta,zeta]])
        
    #NL transformation
    if dP.shape[0] == 12:
        zeta_vector = np.array([1, zeta, zeta, 0.5*zeta**2, zeta**2, 0.5*zeta**2,
                                1, zeta, zeta, 0.5*zeta**2, zeta**2, 0.5*zeta**2])
    
    convergence_parameter = np.sqrt(np.sum((dP*zeta_vector)**2))

    return convergence_parameter


#-------------------------------------------------------------------------------------
def EstimateDisplacementsFourier(F, G, invalid_sub_indices = None):
    """Initial approximation of the deformation as a rigid body translation using
    Fourier analysis.
    
    Convolution theorem: convolution (correlation) in the spatial
    domain is equivalent to multiplication in the frequency domain. The operation is
    generally faster in the frequency domain than in the spatial domain.

    Determine vectors u0, v0: approximate displacement of all subset centers 

    Parameters
    ----------
        F       : Reference image object
        R1       : Deformed image object

    Returns
    -------
        u0      : Approximate X displacement of subset center
        v0      : Approximate Y displacement of subset center
    """

    #initialize vectors to store displacements
    u0 = np.zeros([1,F.sub_centers.shape[1]])
    v0 = np.zeros([1,F.sub_centers.shape[1]])

    #fetch subset size
    sub_size = F.sub_size

    #loop through all subsets, determine the displacements of each subset
    for i in range(0,F.sub_centers.shape[1]):

        if invalid_sub_indices is not None and i in invalid_sub_indices:
                pass
        else:
                #fetch subset center coordinates
                centerx, centery = np.array([F.sub_centers[0][i], F.sub_centers[1][i]])
                
                # print(i)
                # print(F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                #         centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1].shape)
                # print(G.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                        # centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1].shape)
                #extract reference and deformed subset pixel intensity values
                f = F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                        centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
                g = G.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                        centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]

                #normalised cross power spectrum
                cross_power_spec = np.fft.fft2(f)*np.fft.fft2(g).conj()/abs(np.fft.fft2(f)*np.fft.fft2(g).conj())
                #find max CC in frequency domain
                corr_coeff = abs(np.fft.ifft2(cross_power_spec))
                max_CC_index = np.where(corr_coeff == np.max(corr_coeff[:]))

                #fourier shift subset coordinates
                index_shift = -1*np.fft.ifftshift(np.array([np.linspace(-np.fix(sub_size/2),
                                                                        math.ceil(sub_size/2)-1,
                                                                        num=sub_size)]))
                #find displacements based on index positions
                u0[0,i] = index_shift[0,max_CC_index[1]]
                v0[0,i] = index_shift[0,max_CC_index[0]]
        
    return u0, v0


#-------------------------------------------------------------------------------------
def EstimateDisplacementsORB(F, G):
    """Initial approximation of the deformation as a rigid body translation using
    ORB feature detector and iteratively reweighted least squares (IRLS).
    
    Determine vectors u0, v0: approximate displacement of all subset centers 

    Parameters
    ----------
        F       : Reference image object
        G       : Deformed image object

    Returns
    -------
        u0      : Approximate X displacement of subset center
        v0      : Approximate Y displacement of subset center
    """


    #initialize vectors to store displacements
    u0 = np.zeros([1,F.sub_centers.shape[1]])
    v0 = np.zeros([1,F.sub_centers.shape[1]])

    sub_size = F.sub_size
    for i in range(0,F.sub_centers.shape[1]):
        #coordinates of all subset centers
        centerx = F.sub_centers[0,i]
        centery = F.sub_centers[1,i]
        #extract subset based on index i
        f = F.image[centery-int(0.5*(sub_size-1)):centery+int(0.5*(sub_size-1))+1,
                    centerx-int(0.5*(sub_size-1)):centerx+int(0.5*(sub_size-1))+1]

        g = G.image[centery-int(0.5*(sub_size-1)):centery+int(0.5*(sub_size-1))+1,
                    centerx-int(0.5*(sub_size-1)):centerx+int(0.5*(sub_size-1))+1]
        #create ORB object
        descriptor_extractor = ORB(downscale=1.1,n_keypoints=20,fast_n=5,fast_threshold=0.15)
        #f keypoints and descriptors
        descriptor_extractor.detect_and_extract(f)
        keypoints1 = descriptor_extractor.keypoints
        descriptors1 = descriptor_extractor.descriptors
        #g keypoints and descriptors
        descriptor_extractor.detect_and_extract(g)
        keypoints2 = descriptor_extractor.keypoints
        descriptors2 = descriptor_extractor.descriptors
        #image matching
        matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
        #create input data for least-squares
        n_matches = len(matches)
        X = np.zeros((2* n_matches, 4))
        y = np.zeros((2* n_matches, 1))
        for i in range(0, 2* n_matches, 2):
            y[i]   = keypoints2[matches[int(i/2)][1]][0]
            y[i+1] = keypoints2[matches[int(i/2)][1]][1]
            locX = keypoints1[matches[int(i/2)][0]][0]
            locY = keypoints1[matches[int(i/2)][0]][1]
            X[i][0] = 1.0
            X[i][1] = locX
            X[i+1][2] = 1.0
            X[i+1][3] = locY    
        #initialize least-squares
        resRLM = None
        # A robust fit of the model/iteratively reweighted least squares
        modelRLM = sm.RLM( y, X )
        resRLM = modelRLM.fit(maxiter = 20, tol = 1e-4)
        #determine diplcacements from calculated coefficients
        coefficients = resRLM.params
        u0[0,i] = coefficients[0]
        v0[0,i] = coefficients[3]

    return u0, v0


##-------------------------------------------------------------------------------------
def FastInterpolation(image):
    """2D structured Interpolation function
    
    Convert image to a continous domain to retrieve sub-pixel intensity values

    https://github.com/dbstein/fast_interp

    Parameters
    ----------
        image                   : image over which to interpolate
        k                       : order of interpolation function
        (internal parameter)

    Returns
    -------
        image_interpolated      : Relative x coordinates within subsets
    
    """ 

    #image coordinates
    ny = image.shape[0]
    nx = image.shape[1]

        #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    return image_interpolated


##-------------------------------------------------------------------------------------
def NonLinearTrans(G, i):
    """Deform the subset using a Non-Linear polynomial transformation

    Deform the subset using NL transformation coefficients/ shape function
    parameters (SFP's) of the current iteration.

    Parameters
    ----------
        G               : Deformed image object
        i               : Index specifying the XY coordinates of the deformed
                          subset center

    Returns
    -------
        deformed_subset : Deformation of subset at all xy (relative subset) coordinates
    """


    #fetch affine transformation coefficients/shape function parameters (SFP's)
    P = G.P[:,i]

    #fetch subset relative coordinates
    x = G.x
    y = G.y

    #displace, stretch and shear subset in xy-coordinates (14):
    #order of SFP's P[j]: 0,  1,  2,   3,   4,   5,  6,  7,  8,   9,   10,    11
    #                     u   ux  uy  uxx  uxy  uyy  v   vx  vy  vxx   vxy   vyy
    
    deformed_x = 0.5*P[3]*x**2 + P[4]*x*y + 0.5*P[5]*y**2 + (1+P[1])*x + P[2]*y + P[0]
    deformed_y = 0.5*P[9]*x**2 + P[10]*x*y + 0.5*P[11]*y**2 + P[7]*x + (1+P[8])*y + P[6] 
    deformed_subset = np.hstack([deformed_x, deformed_y])

    return deformed_subset


##-------------------------------------------------------------------------------------
def CorrelateImages2D(F, G, settings, invalid_sub_indices = None):
    """Perform correlation between all subsets of two images to determine the SFP's
    that describes the deformation.

    Parameters
    ----------
        F               : Reference image object
        G               : Deformed image object
        settings        : User settings specified in configuration file

    Returns
    -------
        P               : Shape function parameters at convergence (all subsets)
        iterations      : Iterations at convergence (all subsets)
        corr_coeff      : Correlation coefficient at convergence (all subsets)
        stop_val        : Exit criteria at convergence (all subsets)
    """

    #no of subsets for correlation run
    N_subsets = F.sub_centers.shape[1]

    #correlate all subsets
    for i in range(0,N_subsets):
        if invalid_sub_indices is not None and i in invalid_sub_indices:
            pass
        else:
            f, f_mean, f_tilde, dfdx, dfdy  = RefSubsetInfo(F,i)
            #
            Hess, fgrad_X_dWdP = Hessian(dfdx, dfdy, F)
            #check SF order
            if settings['ShapeFunctionOrder'] == 1:
                deltaP =  np.ones([6,1])
            else:
                deltaP = np.ones([12,1])
            itera = 0
            while itera < 100:
                #deform square subset with linear/affine transformation
                #based on current estimation of SFPs
                if settings['ShapeFunctionOrder'] == 1:
                    deformed_subset = AffineTrans(G,i)
                #NL transformation
                else:
                    deformed_subset = NonLinearTrans(G,i)
                
                #extract subset from interpolated G
                g, g_mean, g_tilde = DefSubsetInfo(F, G, deformed_subset, i)
                stop_val = StopCriteria(deltaP, 0.5*(G.sub_size-1))
                if stop_val < 1e-4:
                    break 
                else:
                    Jacobian = np.dot(fgrad_X_dWdP.T, (f[:]-f_mean-f_tilde/g_tilde*(g[:]-g_mean)))
                    deltaP = np.linalg.solve(-Hess, Jacobian) #-1*np.linalg.inv(Hess)*Jacobian.T
                    Pupdate = UpdateSFP(G.P[:,i], deltaP)
                    G.P[:,i:i+1] = Pupdate
                itera = itera + 1
            #Zero-Mean Normalised Cross Correlation Criteria
            G.corr_coeff[0,i] = 1 - sum(((f[:]-f_mean)/f_tilde-(g[:]-g_mean)/g_tilde)**2)/2
            G.stop_val[0,i] = stop_val
            G.iterations[0,i]  = itera


##-------------------------------------------------------------------------------------
def LoadCameraParameters(cam_no):
    """Load camera matrix and distortion coefficients from csv file. Parameters in the
    csv file are those output by running the CalibrateCamera function once.
    The csv files for the camera matrix and distortion coefficients should be loaded
    in the current working directory, or change the path of the source code below.

    Parameters
    ----------
        cam_no          : designation of the camera which has been calibrated

    Returns
    -------
        camera_matrix   : Matrix describing the camera parameters
        dist_coeff      : Coefficients characterising the lens distortion
    """    

    #camera_matrix, dist_coeff

    camera_no = "camera_matrix{}".format(cam_no)
    camera_matrix = open('{}.csv'.format(camera_no), 'rb')
    camera_matrix = loadtxt(camera_matrix ,delimiter = ",")
    camera_matrix = np.array(camera_matrix)

    coeff_no = "dist_coeff{}".format(cam_no)
    dist_coeff = open('{}.csv'.format(coeff_no), 'rb')
    dist_coeff = loadtxt(dist_coeff ,delimiter = ",")
    dist_coeff = np.array(dist_coeff)

    return camera_matrix, dist_coeff


#-------------------------------------------------------------------------------------
def CalibrateCamera(cam_no):
    """Perform camera calibration using a set of images containg a calibration plate
    with distinct calibration targets. This function is currently setup  to calibrate
    the plate supplied by LaVision. The plat has circular targets and two target planes.
    
    Load the calibration images in a folder named 'calibration_images' in the current
    working directory or modify the source code below.

        Parameters
        ----------
            cam_no          : Desgination of the camera to be calibrated. The designation
                              is just a number that identifies the folder number in which
                              to search for the images of the calibration plate.

        Returns
        -------
            camera_matrix   : Matrix describing the camera parameters
            dist_coeff      : Coefficients characterising the lens distortion

    """

    #Specify the shape and number of grid to be detected in pattern image
    CIRCLESGRID = (11, 11)
    #convergence criteria:
    #no of iterations
    #change between iterations is sufficiently small (epsilon)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #Vector for 3D, 2D points
    threeD_points = []
    twoD_points = []

    #vector for real world coordinates
    objectp3d = np.zeros((1, CIRCLESGRID[0]
            * CIRCLESGRID[1],
            3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CIRCLESGRID[0],
                0:CIRCLESGRID[1]].T.reshape(-1, 2)
    #flag
    prev_img_shape = None

    #SimpleBlobDetector parameters

    # 'BLOB' detector - looking for circles in the pattern
    blobParams = cv.SimpleBlobDetector_Params()

    #blob detector parameters/filters
    #Thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 255
    #Area
    blobParams.filterByArea = True
    blobParams.minArea = 64  
    blobParams.maxArea = 2500  
    #Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.6
    #Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87
    #Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01
    #Create a detector with the parameters
    blobDetector = cv.SimpleBlobDetector_create(blobParams)

    #find all images in specified directory
    images = glob.glob('calibration_images{}/*.tif'.format(cam_no))
    for filename in images:
        #read in image
        im = cv.imread(filename)
        im2 = 1- im
        
        #find circles in pattern
        keypoints = blobDetector.detect(im2) 
        im_with_keypoints = cv.drawKeypoints(im2, keypoints, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_gray = cv.cvtColor(im_with_keypoints, cv.COLOR_BGR2GRAY)
        
        #find the grid of circles in the pattern
        ret, corners = cv.findCirclesGrid(im_with_keypoints, (11,11), None, flags = cv.CALIB_CB_SYMMETRIC_GRID)
        #if pattern is detected, save the relevant coordinates and display their detected grid
        #count number of good images
        N_images = 0
        if ret == True:
            corners2 = corners
            im_with_keypoints2 = copy.deepcopy(im_with_keypoints)
            #store points in relevant vectors
            threeD_points.append(objectp3d)
            twoD_points.append(corners)
            image = cv.drawChessboardCorners(im_with_keypoints2, (11,11), corners2, ret)
            N_images = N_images + 1


        #display detected grid in pattern, press enter to continue to next calibration image in directory
        cv.imshow('img', image)
        cv.waitKey(0)

        cv.destroyAllWindows()
        h, w = image.shape[:2]
        ret, camera_matrix, dist_coeff, r_vecs, t_vecs = cv.calibrateCamera(threeD_points, twoD_points, im_with_keypoints_gray.shape[::-1], None, None) 
        
    #save results to csv file for re-use
    np.savetxt("camera_matrix{}.csv".format(cam_no), camera_matrix, delimiter = ",")
    np.savetxt("dist_coeff{}.csv".format(cam_no), dist_coeff, delimiter = ",")

    return camera_matrix, dist_coeff, N_images


#-------------------------------------------------------------------------------------
def SIFTquick(F1_ROI_padded, F1, F2):
    """Find keypoints and descriptors in the reference images of the two cameras.
    SIFT is used since it finds a dense population of keypoints in an image in comparison
    to ORB.

    Parameters
    ----------
        F1_ROI_padded   :   Padded region of interest in reference image of camera 1
        F1, F2          :   Reference image objects of cameras 1&2

    Returns
    -------
        Xk1, Xk2        :   Numpy arrays of keypoints (before matching is performed)
        Xo1             :   Subset centre coordinates in reference image of camera 1
        N_subsets       :   No of subsets in ROI
        kp1, kp2        :   Detected keypoints in reference images of cameras 1&2 (cvObject)
                            (before matching is performed) 
        d1, d2          :   Descriptors of detected keypoints in reference images of
                            cameras 1&2 (cvObject)

    """

    #keypoint matching/feature detection between reference images of from both cameras
    #create SIFT feature detector object
    sift = cv.xfeatures2d.SIFT_create()
    #find keypoints and descriptors in both images
    kp1, d1 = sift.detectAndCompute(F1_ROI_padded,None)
    kp2, d2 = sift.detectAndCompute(F2.image_8bit,None)

    #cast keypoints to numpy arrays for ease of use
    #F1 keypoints x and y coordinates
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

    #F1 subset centre coordinates
    xo1 = F1.sub_centers[0,:]
    yo1 = F1.sub_centers[1,:]
    Xo1 = F1.sub_centers
    #number of subsets
    N_subsets = F1.sub_centers.shape[1]
    #subset halfwidth
    hw = (F1.sub_size-1)/2

    return Xk1, Xk2, Xo1, N_subsets, kp1, kp2, d1, d2


#-------------------------------------------------------------------------------------
def KeypointMatchRobust(Xk1, Xk2, kp1, kp2, d1, d2, F1_ROI_padded, F2):
    """Perform keypoint, descriptor matching between keypoints in reference images 1&2.

    This function has been set up to have a high accuracy of keypoint matches and
    high computational speed.

    Parameters
    ----------
        Xk1, Xk2        :   Numpy arrays of keypoints before matching is performed
        kp1, kp2        :   Detected keypoints in reference images of cameras 1&2 (cvObject) 
        d1, d2          :   Descriptors of detected keypoints in reference images of
                            cameras 1&2 (cvObject)
        F1_ROI_padded   :   Padded region of interest in reference image of camera 1
        F2              :   Reference image object of camera 2
        

    Returns
    -------
        Xk1, Xk2        :   Numpy arrays of matching keypoints after bad matches
                            have been eliminated

    """


    #find matches between image 1 and 2 with knnmatch and identify good matches
    bf = cv.BFMatcher()
    #store the indices of the matches in F1 and F2
    matches = bf.knnMatch(d1, d2, k=2)
    #only keep good matches using condition inisde for loop
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append([m])
    img3 = cv.drawMatchesKnn(F1_ROI_padded, kp1, F2.image_8bit, kp2, good_matches, None, flags=2)
    plt.imshow(img3),plt.show()

    #cast match indices to numpy array for knnMatch
    N_matches = np.size(good_matches)
    p_matches = np.zeros([N_matches, 2])
    for j in range(0, N_matches):
        #F1 descriptors
        p_matches[j, 0] = good_matches[j][0].queryIdx
        #F2 descriptors
        p_matches[j, 1] = good_matches[j][0].trainIdx

    #remove bad matches from feature detection results
    Xk1 = Xk1[:, p_matches[:,0].astype(int)]
    Xk2 = Xk2[:, p_matches[:,1].astype(int)]

    return Xk1, Xk2


#-------------------------------------------------------------------------------------
def kNNSubCentres(N_subsets, Xo1, Xk1, k = 20):
    """Find the k Nearest Neighbours/keypoints to the subset centres in the reference
    image of camera 1.

    Parameters
    ----------
        k               :   No. of nearest neighbours/keypoints in vicinity of subset
                            centre to locate 
        N_subsets       :   Number of subsets in ROI
        Xk1             :   Keypoint coordinates in reference image of camera 1
        Xo1             :   Subset centre coordinates in reference image of camera 1

    Returns
    -------
        kNN_indices     :   Indices of the k nearset keypoints to each respective
                            subset centre. The indices are the IDs for keypoints in
                            the keypoints vector

    """
    #identify K nearest keypoints(neighbours) to each subset centre
    kNN_indices = np.zeros([N_subsets,k])

    for i in range(0, N_subsets):
        #SSD between subset centre i and all keypoints
        diff = np.sqrt((Xo1[0,i]-Xk1[0,:])**2 + (Xo1[1,i]-Xk1[1,:])**2)
        idx = np.argpartition(diff, k)
        #k minimum SSD between subset centre i and all keypoints
        kNN_indices[i, :] = idx[:k]

    return kNN_indices


#-------------------------------------------------------------------------------------
def MatchSubCentresProjective(F1, F2, N_subsets, Xk1, Xk2, kNN_indices):
    """Given two reference images taken from different views with their respective cameras.
    The coordinates of the subset centres in the first reference image need to be identified
    in the second reference image. This function determines an initial estimate of the
    relationship between the two views using an affine transformation.

    Parameters
    ----------
        F1              :   Reference image from first camera image series
        N_subsets       :   Number of subsets in ROI
        Xk1, Xk2        :   Keypoint coordinates in reference images of camera 1&2
        kNN_indices     :   k Nearest keypoints to each respective subset centre

    Returns
    -------
        P0_stereo       :   Initial estimate of affine transformation parameters describing
                            relationship between reference images of the two cameras

        """
    #loop through all subsets, fit projective transformation model between left and right images using RANSAC
    centers = np.zeros([F2.sub_centers.shape[0], F2.sub_centers.shape[1]])

    for j in range(0, N_subsets):
        #fetch centre coordinates of current subset in F1 (reference image 1)
        xo1 = F1.sub_centers[0, j]
        yo1 = F1.sub_centers[1, j]

        #find kNN keypoint coordinates for current subset
        xkp1 = Xk1[:,kNN_indices[j,:].astype(int)].T
        xkp2 = Xk2[:,kNN_indices[j,:].astype(int)].T
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

        #subset centre coordinates of F2 in sensor coordinates
        xo2 = (h11*xo1 + h12*yo1 + h13)/(h31*xo1 + h32*yo1 + h33)
        yo2 = (h21*xo1 + h22*yo1 + h23)/(h31*xo1 + h32*yo1 + h33)

        #store SFP's 
        #F2.sub_centers[:, j] = xo2, yo2
        centers[0, j] = xo2
        centers[1, j] = yo2
        centers = np.floor(centers).astype('int')

    #locate the subset centres in reference image 2 based on matching above
    F2.sub_centers = centers
    
    return F2.sub_centers


#-------------------------------------------------------------------------------------
def FundamentalMatrixEstORB():
    """The fundamental matrix describes the relationship between any two images
    of the same scene that constrains where the projection of points from the scene can
    occur in both images.
    https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
    
    This function uses the reference (first/undeformed) images from the two image series
    (taken by two different cameras configured in a stereo setup) to peform the fundamental
    matrix estimation.

    The feature detector used here is ORB. This results in a sparse distribution of keypoints
    to be matched. Outlier keypoints are eliminated using RANSAC. For a more accurate
    estimation of the fundamental matrix a denser distribution of keypoints are recommended,
    using a feature detector such as SIFT.
 

        Returns
        -------
            Fu              : Fundamental matrix 

    """
    
    img_left = cv.imread('images/R0.tif', 0)
    img_right = cv.imread('images/L0.tif', 0)

    # Find sparse feature correspondences between left and right image.
    #descriptors and matches
    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img_left)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_right)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                            cross_check=True)

    # Estimate the epipolar geometry between the left and right image.
    random_seed = 9
    rng = np.random.default_rng(random_seed)

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                            keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=0.6, max_trials=5000,
                            random_state=rng)

    #fundamental matrix
    Fu = model.params

    return Fu


#-------------------------------------------------------------------------------------
def MatchSubsetCentres(F1, F1_ROI_padded, F2):
    """Given two reference images taken from different views with their respective cameras.
    The coordinates of the subset centres in the first reference image need to be identified
    in the second reference image.

    Parameters
    ----------
        F1_ROI_padded   : Measurement region of interest (ROI) defined in the first reference
                          image frame. The ROI is populated with subsets and defines the
                          relevant area for feature detection.
        F1              : Reference image from the first camera/image series
        F2              : Reference image from the second camera/image series

    Returns
    -------
        P0_stereo       : SFPs describing the relationship between the views of the
                          reference images. The displacement components of P0_stereo are
                          used to locate the subset centre coordinates in the second reference
                          image.
    """

    
    #feature detection, keypoints and descriptors
    Xk1, Xk2, Xo1, N_subsets, kp1, kp2, d1, d2 = SIFTquick(F1_ROI_padded, F1, F2)
    #robust matching of keypoints between images
    Xk1, Xk2 = KeypointMatchRobust(Xk1, Xk2, kp1, kp2, d1, d2, F1_ROI_padded, F2)
    #find k nearest keypoints (neighbours) to each subset centre
    kNN_indices = kNNSubCentres(N_subsets, Xo1, Xk1, k = 20)
    #describe relationship between subset centre coordinates
    #between images using projective transformation model
    F2_sub_centers = MatchSubCentresProjective(F1, F2, N_subsets, Xk1, Xk2, kNN_indices) 

    return F2_sub_centers


#-------------------------------------------------------------------------------------
def Triangulate(Fu, Q1, Q2, X1, X2, invalid_sub_indices):
    """Determine out of plane displacments through triangulation of image coordinates 
    from the two cameras in the stereo setup. In the program the function is used to 
    find the 3D coordinates of the subset centres before and after deformation, the
    displacement is the difference of this.

    Parameters
    ----------
        Fu              : Fundamental matrix describing the relationship of the two
                          camera views
        Q1, Q2          : Augmented camera matrices of Kj's (the internal camera parameter
                          matrices)
        X1, X2          : Displacement/coordinates from the two camera views

    Returns
    -------
       X_3D             : 3D world displacement/coordinates vector
    """
    
    #number of points to consider
    N_points = X1.shape[1]
    X_3D = np.zeros([3, N_points])
    #loop through points of interest
    for l in range(0,N_points):
        if l in invalid_sub_indices:
            pass
        else:
            #translation matrices, translates undistorted sensor coordinates
            #to the origin of its coordinate system
            T1inv = np.array([
                                [1, 0, X1[0,l]],
                                [0, 1, X1[1,l]],
                                [0, 0,   1    ]
                                ]).astype('double')
            #
            T2inv = np.array([
                                [1, 0, X2[0,l]],
                                [0, 1, X2[1,l]],
                                [0, 0,   1    ]
                                ]).astype('double')
                                
            #updated fundamental matrix
            Fu1 = T2inv.T.dot(Fu).dot(T1inv)
            #determine epipoles of two cameras
            #SVD
            U, _, V = la.svd(Fu1)
            V = V.T

            #normalisation of epipoles
            e1 = V[:,2]/la.norm(V[0:2, 2])
            e2 = U[:,2]/la.norm(U[0:2, 2])
            #rotation matrices, in order to place the epipoles on the x-axes
            #of the respective coordinate systems 
            R1 = np.array([[e1[0], e1[1], 0], [-e1[1], e1[0], 0], [0, 0, 1]])
            R2 = np.array([[e2[0], e2[1], 0], [-e2[1], e2[0], 0], [0, 0, 1]])
            #updated fundamental matrix
            Fu2 = R2.dot(Fu1).dot(R1.T)

            #elments of fundamental matrix to be used in triangulation polynomial cost function
            phi1 = Fu2[1,1]
            phi2 = Fu2[1,2]
            phi3 = Fu2[2,1]
            phi4 = Fu2[2,2]
            #coefficients of polynomial to be minimised
            p = np.array([
                            (-phi4*phi1**2*phi3*e1[2]**4 + phi2*phi1*phi3**2*e1[2]**4),

                            (phi1**4 + 2*phi1**2*phi3**2*e2[2]**2 - phi1**2*phi4**2*e1[2]**4 +
                                phi2**2*phi3**2*e1[2]**4 + phi3**4*e2[2]**4),

                            (4*phi1**3*phi2 - 2*phi1**2*phi3*phi4*e1[2]**2 +
                            4*phi1**2*phi3*phi4*e2[2]**2 + 2*phi1*phi2*phi3**2*e1[2]**2 +
                            4*phi1*phi2*phi3**2*e2[2]**2 - phi1*phi2*phi4**2*e1[2]**4 + 
                            phi2**2*phi3*phi4*e1[2]**4 + 4*phi3**3*phi4*e2[2]**4),

                            (6*phi1**2*phi2**2 - 2*phi1**2*phi4**2*e1[2]**2 + 
                            2*phi1**2*phi4**2*e2[2]**2 + 8*phi1*phi2*phi3*phi4*e2[2]**2 + 
                            2*phi2**2*phi3**2*e1[2]**2 + 2*phi2**2*phi3**2*e2[2]**2 +
                            6*phi3**2*phi4**2*e2[2]**4),

                            (-phi1**2*phi3*phi4 + 4*phi1*phi2**3 + phi1*phi2*phi3**2 - 
                            2*phi1*phi2*phi4**2*e1[2]**2 + 4*phi1*phi2*phi4**2*e2[2]**2 + 
                            2*phi2**2*phi3*phi4*e1[2]**2 + 4*phi2**2*phi3*phi4*e2[2]**2 +
                            4*phi3*phi4**3*e2[2]**4),

                            (-phi1**2*phi4**2 + phi2**4 + phi2**2*phi3**2 + 
                            2*phi2**2*phi4**2*e2[2]**2 + phi4**4*e2[2]**4),

                            (phi3*phi2**2*phi4 - phi1*phi2*phi4**2)   
                        ])

            #roots of the polynomial, coefficients stored in p
            p_roots = np.roots(p)
            #keep roots with no imaginary part
            p_roots = p_roots[p_roots.imag == 0]
            p_roots = p_roots.real

            #geometric error cost function to be minimised
            #evlatuated 
            Ds = p_roots**2/(1+(p_roots*e1[2])**2) + (phi3*p_roots + phi4)**2/((phi1*p_roots + phi2)**2 +e2[2]**2*(phi3*p_roots + phi4)**2)
            #optimal value
            t  = np.min(Ds)
            #sensor coordinates converted back to euclidian space, 
            #in original sensor coordinate system
            X1_ = T1inv.dot(R1.T).dot(np.array([[t**2*e1[2]], [t], [t**2*e1[2]**2 + 1]]))
            X2_ = T2inv.dot(R2.T).dot(np.array([[e2[2]*(phi3*t + phi4)**2],
                                        [-(phi1*t + phi2)*(phi3*t + phi4)],
                                        [(phi1*t + phi2)**2 + e2[2]**2*(phi3*t + phi4)**2]]))

            #removal of scaling variable applied for homogeneous coordinates
            X1_out = X1_[0:2]/X1_[2]
            X2_out = X2_[0:2]/X2_[2]
            #matrix to decompose below
            S = np.array([X1_out[0][0]*Q1[2,:] - Q1[0,:],
                                X1_out[1][0]*Q1[2,:] - Q1[1,:],
                                X2_out[0][0]*Q2[2,:] - Q2[0,:],
                                X2_out[1][0]*Q2[2,:] - Q2[1,:]])

            #SVD
            _, _, V = la.svd(S)
            V = V.T

            #world coordinates
            X_3D[:, l] = V[0:3,3]/V[3,3]

    return X_3D


#-------------------------------------------------------------------------------------
def UndistortPoints(X, camera_matrix, dist_coeff):
    """Undistort subset centre coordinates from ideal sensor coordinate system to real/
    undistorted sensor coordinate system.

    Parameters
    ----------
        X                 : Displacement vector in ideal camera coordinate system
        camera_matrix(K)  : Internal camera parameter matrix determined by calibration
        dist_coeff        : Distortion coefficients of camera system determined
                            by calibration

    Returns
    -------
        X_undistorted      : Undistorted displacements in camera coordinate system
    """
    #number of coordinates to undistort
    n_points = X.shape[0]
    #reshape coordinates to appropriate shape for cv.undistortpoints function
    X_ = np.zeros((n_points,1,2), dtype=np.float32)
    X_[:, 0, :] = X[:, :]
    # do the actual undistort
    X_undistorted = cv.undistortPoints(X_,camera_matrix, dist_coeff, P=camera_matrix)

    return X_undistorted[:, 0, :]
































#INCOMPLETE FUNCTIONS (NOT IN  USE)
#------------------------------------------------------------------------------------- 
class DIC_2D_Subset:

    def __init__(self, image_set, settings):
        self.settings = settings
        self.image_set = image_set
        self.results = CorrelateImages(image_set, settings)

    def Summary():
        pass
    def PlotDisplacements():
        pass
    def PlotStrain():
        pass
    def PlotStress():
        pass
    

#-------------------------------------------------------------------------------------        
class StoreDICResults:

    def __init__(self, displacement):
        
        self.displacement = displacement


#-------------------------------------------------------------------------------------
class PlotDICResults(StoreDICResults):

    def __init__(self, displacement):

        super().__init__(displacement)
        #self.meshgrid =


#-------------------------------------------------------------------------------------
def SettingsInfo():

#return a summary of the correlation and image setup
    pass


#-------------------------------------------------------------------------------------
def RegularizeDisplacements():
    pass


def RelativeCameraPose(E):
    #SVD of essential matrix
    U, S, V = np.linalg.svd(E)
    
    #matrix for composition to determine relative translation 
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    #matrix for composition to determine relative orientation 
    Z = np.array([[0,  1, 0],
                  [-1, 0, 0],
                  [0,  0, 0]])

    #scaling factor for relative translation vector
    k = S[0,0]
    #skew-symmetric translation matrix
    Tx = k*U.dot(Z).dot(U.T)
    #R =



    #deteremine relative camera orientation and translation from SVD of essential matrix

    pass


#-------------------------------------------------------------------------------------
def MatchSubCentresAffine(F1, N_subsets, Xk1, Xk2, kNN_indices):
    """Given two reference images taken from different views with their respective cameras.
    The coordinates of the subset centres in the first reference image need to be identified
    in the second reference image. This function determines an initial estimate of the
    relationship between the two views using an affine transformation.

    Parameters
    ----------
        F1              :   Reference image from first camera image series
        N_subsets       :   Number of subsets in ROI
        Xk1, Xk2        :   Keypoint coordinates in reference images of camera 1&2
        kNN_indices     :   k Nearest keypoints to each respective subset centre

    Returns
    -------
        P0_stereo       :   Initial estimate of affine transformation parameters describing
                            relationship between reference images of the two cameras

        """
    #loop through all subsets, fit affine model between left and right images using RANSAC
    P0_stereo = np.zeros([F1.P.shape[0], F1.P.shape[1]])

    for j in range(0, N_subsets):
        #fetch centre coordinates of current subset in F1 (reference image 1)
        xo1 = F1.sub_centers[0, j]
        yo1 = F1.sub_centers[1, j]

        #find kNN keypoint coordinates for current subset
        xkp1 = Xk1[:,kNN_indices[j,:].astype(int)].T
        xkp2 = Xk2[:,kNN_indices[j,:].astype(int)].T
        model_robust, inliers = ransac((xkp1, xkp2), AffineTransform, min_samples=3,
                                residual_threshold=2, max_trials=100)
        
        #coefficients of the affine transformation
        a1 = model_robust.params[0][0] - 1
        a2 = model_robust.params[0][1]
        a3 = model_robust.params[0][2]
        a4 = model_robust.params[1][0]
        a5 = model_robust.params[1][1] - 1
        a6 = model_robust.params[1][2]

        #SFP's describing the mapping between the j'th subset in the two cameras,
        #based on affine transformation models
        u = a1*xo1 + a2*yo1 + a3
        ux = a1
        uy = a2
        uxx = 0
        uxy = 0
        uyy = 0
        v = a4*xo1 + a5*yo1 + a6
        vx = a4
        vy = a5
        vxx = 0 
        vxy = 0
        vyy = 0

        #store SFP's 
        P0_stereo[:, j] = u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy
    
    return P0_stereo














































#-------------------------------------------------------------------------------------
# def CreateSubsetsSample14(setup):
#     """Create image subsets
    
#     Store XY coordinates of each subset center
#     (XY coordinates are the global coordinates in the mother image)
    
#     Store xy coordinates of pixels within subset, these coordinates are the same
#     for all subsets
#     (xy coordinates are the relative local coordinates within the subset)


#     Parameters
#     ----------
#         setup       : [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
#                        sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    
#     Returns
#     -------
#         x           : Relative x coordinates within subsets
#         y           : Relative y coordinates within subsets
#         sub_centers : XY coordinates of all subsets' centers
#     """

#     #fetch setup variables
#     sub_size = int(setup[0])
#     sub_frequency = int(setup[5])
#     img_rows = int(setup[6]) 
#     img_columns = int(setup[7])

#     #create subset centers XY coordinates as MESHGRIDS
#     sub_centers_y, sub_centers_x = np.meshgrid(np.arange(25,
#                                                          570,
#                                                          sub_frequency),
#                                                np.arange(25,
#                                                          2030,
#                                                          sub_frequency),
#                                                indexing = 'ij')


#     #

#     #flatten subset centers XY coordinates to vectors
#     sub_centers_x = np.array([sub_centers_x.flatten(order = 'F')]).T
#     sub_centers_y = np.array([sub_centers_y.flatten(order = 'F')]).T
#     sub_centers = np.vstack((sub_centers_x.T,sub_centers_y.T))

#     #create subset xy relative coordinates
#     [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
#                          np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
#                          indexing='ij')

#     #flatten subset xy relative coordinates to vectors
#     x = np.array([x.flatten(order = 'F')]).T
#     y = np.array([y.flatten(order = 'F')]).T

#     return x, y, sub_centers


#-------------------------------------------------------------------------------------
# def CreateSubsets(settings):
#     """Create image subsets
    
#     Store XY coordinates of each subset center
#     (XY coordinates are the global coordinates in the mother image)
    
#     Store xy coordinates of pixels within subset, these coordinates are the same
#     for all subsets
#     (xy coordinates are the relative local coordinates within the subset)


#     Parameters
#     ----------
#         setup       : [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
#                        sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    
#     Returns
#     -------
#         x           : Relative x coordinates within subsets
#         y           : Relative y coordinates within subsets
#         sub_centers : XY coordinates of all subsets' centers
#     """

#     #fetch setup variables
#     sub_size = int(settings[0])
#     sub_frequency = int(settings[5])
#     img_rows = int(settings[6]) 
#     img_columns = int(settings[7])

#     #create subset centers XY coordinates as MESHGRIDS
#     sub_centers_y, sub_centers_x = np.meshgrid(np.arange(int(0.5*(sub_size-1)+sub_frequency),
#                                                          int(img_rows-0.5*(sub_size-1)),
#                                                          sub_frequency),
#                                                np.arange(int(0.5*(sub_size-1)+sub_frequency),
#                                                          int(img_columns-0.5*(sub_size-1)),
#                                                          sub_frequency),
#                                                indexing = 'ij')

#     #flatten subset centers XY coordinates to vectors
#     sub_centers_x = np.array([sub_centers_x.flatten(order = 'F')]).T
#     sub_centers_y = np.array([sub_centers_y.flatten(order = 'F')]).T
#     sub_centers = np.vstack((sub_centers_x.T,sub_centers_y.T))

#     #create subset xy relative coordinates
#     [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
#                          np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
#                          indexing='ij')

#     #flatten subset xy relative coordinates to vectors
#     x = np.array([x.flatten(order = 'F')]).T
#     y = np.array([y.flatten(order = 'F')]).T

#     return x, y, sub_centers