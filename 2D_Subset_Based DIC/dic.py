"""Subset-based 2D Digital Image Correlation library

    Functions in this library are used to perform Subset-based 2D-DIC.

    Notes
    -----
    This is an open-source library.

    This code was setup by Ed Brisley 
    June 30, 2023
    
    References
    ----------
    [1] github repository link: 
"""
##-------------------------------------------------------------------------------------
# Importing libraries
#-------------------------------------------------------------------------------------
from tokenize import Double
import numpy as np
import matplotlib.pyplot as plt
import math
import importlib
import sys  
import copy
import scipy as sp
import csv
import cv2 as cv
from scipy.linalg import solve
import scipy.interpolate
import time
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import skimage as sk
from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray
import statsmodels.api as sm
import patsy as ps
from patsy import dmatrices
from patsy import dmatrix
from numpy import loadtxt
import os
from configparser import ConfigParser


#-------------------------------------------------------------------------------------
# Versioning the code
# 0.1.0 - 
# Version 0.2.0 - 
# Version 0.2.1 - 
#-------------------------------------------------------------------------------------
__version__="0.2.1"



#-------------------------------------------------------------------------------------
def LoadSettings(image_set):
    ##load the configuration file containing the DIC settings
    configur = ConfigParser()
    configur.read('Settings.ini')
    sub_size = configur.getint('Subsets','SubsetSize')
    sub_frequency = configur.getint('Subsets','SubsetFrequency')
    GF_stddev = configur.getfloat('Filters','GaussianFilterStdDev')
    GF_filtsize = configur.getint('Filters','GaussianFilterSize')
    SF_order = configur.getint('Miscellaneous','ShapeFunctionOrder')
    corr_refstrat = configur.getint('Miscellaneous','CorrelationReferenceStrategy')
    calibrate = configur.getint('Miscellaneous','Calibration')
    #store setup parameters
    settings = np.array([])
    settings = np.append(settings,sub_size)
    settings = np.append(settings,SF_order)
    settings = np.append(settings,GF_stddev)
    settings = np.append(settings,GF_filtsize)
    settings = np.append(settings,corr_refstrat)
    settings = np.append(settings,sub_frequency)
    #define images working directory
    img_0 = cv.imread("images/{}".format(image_set[0]),0)
    img_rows = img_0.shape[0]
    img_columns = img_0.shape[1]
    settings = np.append(settings,img_rows)
    settings = np.append(settings,img_columns)
    settings = np.append(settings,corr_refstrat)
    settings = np.append(settings,calibrate)
    return settings


#-------------------------------------------------------------------------------------
def LoadImages():
    ##load the images from directory
    current_working_directory = os.getcwd()
    folder = '\images'
    image_location = current_working_directory + folder
    image_set = []
    for filename in os.listdir(image_location):    
        image_set.append(filename)
    return image_set


#-------------------------------------------------------------------------------------
def SettingsInfo():
#return a summary of the correlation and image setup
    pass


#-------------------------------------------------------------------------------------
class DICImage:
    def __init__(self,image,settings):
        self.image = cv.normalize(image.astype('double'), None, 0.0, 1.0, cv.NORM_MINMAX)
        self.sub_size =  int(settings[0])
        self.SF_order = settings[1]
        self.GF_stddev = settings[2]
        self.GF_filtsize = int(settings[3])
        self.corr_refstrat = settings[4]
        self.frequency = int(settings[5])
        self.img_rows = settings[6]
        self.img_columns = settings[7]
        #create coordinates of pixels in subset relative to centre, dX dY
        #(constant subset size-these relative coordinates are the same for all subsets)
        #create coordinates for subset centers
        self.x, self.y, self.sub_centres = CreateSubsets(settings)
        #shape function parameters of F, 12 parameters for each subset
        self.P = np.zeros([6, self.sub_centres.shape[1]]) 
        self.sub_halfwidth = (self.sub_size-1)/2


#-------------------------------------------------------------------------------------       
class ReferenceImage(DICImage):
    def __init__(self,image,settings):
        super().__init__(image,settings)
        #gaussian filter applied to reference image
        self.image = cv.GaussianBlur(self.image,
                                    (self.GF_filtsize,self.GF_filtsize),
                                    self.GF_stddev)
        #gradient of F in xy directions
        grad = np.array(np.gradient(self.image),dtype= float)
        self.F_grady = grad[0]
        self.F_gradx = grad[1]


#-------------------------------------------------------------------------------------
class DeformedImage(DICImage):
    def __init__(self,image,settings):
        super().__init__(image,settings)
        #add blurring to 
        self.image = cv.GaussianBlur(self.image,
                                    (self.GF_filtsize,self.GF_filtsize),
                                    self.GF_stddev)
        #interpolation of deformed image
        self.G_interpolated = scipy.interpolate.RegularGridInterpolator(
                              (np.linspace(0,self.image.shape[0]-1,self.image.shape[0]),
                               np.linspace(0,self.image.shape[1]-1,self.image.shape[1])),
                               self.image)
        
        #variables to save correlation run results at convergence
        self.corr_coeff = np.zeros([1,self.sub_centres.shape[1]])
        self.stop_val = np.zeros([1,self.sub_centres.shape[1]])
        self.iterations = np.zeros([1,self.sub_centres.shape[1]])


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
def CreateSubsets(setup):
    """Create image subsets
    
    Store XY coordinates of each subset centre
    (XY coordinates are the global coordinates in the mother image)
    
    Store xy coordinates of pixels within subset, these coordinates are the same
    for all subsets
    (xy coordinates are the relative local coordinates within the subset)


    Parameters
    ----------
        setup       : [sub_size, SF_order, GF_stddev, GF_filtsize, corr_refstrat,
                       sub_frequency, img_rows, img_columns, corr_refstrat, calibrate]
    
    Returns
    -------
        x           : Relative x coordinates within subsets
        y           : Relative y coordinates within subsets
        sub_centers : XY coordinates of all subsets' centres
    """

    #fetch setup variables
    sub_size = int(setup[0])
    sub_frequency = int(setup[5])
    img_rows = int(setup[6]) 
    img_columns = int(setup[7])

    #create subset centres XY coordinates as MESHGRIDS
    sub_centres_y, sub_centres_x = np.meshgrid(np.arange(int(0.5*(sub_size-1)+sub_frequency),
                                                         int(img_rows-0.5*(sub_size-1)),
                                                         sub_frequency),
                                               np.arange(int(0.5*(sub_size-1)+sub_frequency),
                                                         int(img_columns-0.5*(sub_size-1)),
                                                         sub_frequency),
                                               indexing = 'ij')

    #flatten subset centres XY coordinates to vectors
    sub_centres_x = np.array([sub_centres_x.flatten(order = 'F')]).T
    sub_centres_y = np.array([sub_centres_y.flatten(order = 'F')]).T
    sub_centers = np.vstack((sub_centres_x.T,sub_centres_y.T))

    #create subset xy relative coordinates
    [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                         np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                         indexing='ij')

    #flatten subset xy relative coordinates to vectors
    x = np.array([x.flatten(order = 'F')]).T
    y = np.array([y.flatten(order = 'F')]).T

    return x, y, sub_centers


#-------------------------------------------------------------------------------------
def RefSubsetInfo(F,i):
    """Extract reference subset, f, from reference image, F

    Extract subset f from F, based on the subset centre coordinates of f - specified by
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

    #fetch XY coordinates of i'th subset centre
    centerx, centery = np.array([F.sub_centres[0][i],F.sub_centres[1][i]])

    #extract  refrence subset intensity values, f, from mother image, F,
    #based on subset center coordinates
    f = F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]

    #extract subset spatial gradients
    dfdy = F.F_grady[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
    dfdx = F.F_gradx[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]

    #average subset intensity, and normalsied sum of squared differences
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    f = np.array([f.flatten(order = 'F')]).T

    return f, f_mean, f_tilde, dfdx, dfdy


#-------------------------------------------------------------------------------------
def Hessian(dfdx,dfdy,F):
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

    #procuct of subset gradient [dfdx, dfdy] and warp function gradient dWdP ?? 
    fgrad_X_dWdP = np.array([
                             dfdx[:,0]*1,
                             dfdx[:,0]*x[:,0],
                             dfdx[:,0]*y[:,0],
                             dfdy[:,0]*1,
                             dfdy[:,0]*x[:,0],
                             dfdy[:,0]*y[:,0]
                             ]).T

    #compute hessian
    Hess = np.dot(fgrad_X_dWdP.T, fgrad_X_dWdP)   
    return Hess, fgrad_X_dWdP


#-------------------------------------------------------------------------------------
def AffineTrans(G,i):
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
def DefSubsetInfo(G, deformed_subset,i):
    """Extract deformed subset, g, from deformed image, G

    Extract subset g from G, based on the subset centre coordinates of g - specified 
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

    #fetch XY coordinates of i'th subset centre
    centerx, centery = np.array([G.sub_centres[0][i],G.sub_centres[1][i]])

    #add deformed xy coordinates (determined using AffineTransform)
    #to the XY coordinates of the subset center
    #this gives the XY coordinates of the deformed subset in the mother image
    Y = centery*np.ones(N_points) + deformed_subset[:,1]
    X = centerx*np.ones(N_points) + deformed_subset[:,0]

    #extract  deformed subset intensity values, g, from mother image, G.Interpolated,
    #based on deformed subset XY coordinates
    g = np.zeros([N_points,1])
    for m in range(0,N_points):
        g[m] = G.G_interpolated(np.array([Y[m],X[m]]))

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

    #(22)
    #w of current estimate of SFPs
    #order of SFP's P[i]: 0,  1,  2,  3,  4,  5
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

    #(21)
    #extract updated SFP's from P update/up matrix
    Pupdate = np.array([
                        [up[0,2]],
                        [up[0,0]-1],
                        [up[0,1]],
                        [up[1,2]],
                        [up[1,0]],
                        [up[1,1]-1]
                       ])

    return Pupdate


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
    # (23)
    #create zeta vector and compute convergence_parameter
    zeta_vector = np.array([[1,zeta,zeta,1,zeta,zeta]])
    convergence_parameter = np.sqrt(np.sum((dP*zeta_vector)**2))

    return convergence_parameter


#-------------------------------------------------------------------------------------
def EstimateDisplacementsFourier(F,G):
    """Initial approximation of the deformation as a rigid body translation using
    Fourier analysis.
    
    Convolution theorem: convolution (correlation) in the spatial
    domain is equivalent to multiplication in the frequency domain. The operation is
    generally faster in the frequency domain than in the spatial domain.

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
    u0 = np.zeros([1,F.sub_centres.shape[1]])
    v0 = np.zeros([1,F.sub_centres.shape[1]])

    #fetch subset size
    sub_size = F.sub_size

    #loop through all subsets, determine the displacements of each subset
    for i in range(0,F.sub_centres.shape[1]):

        #fetch subset centre coordinates
        centerx, centery = np.array([F.sub_centres[0][i],F.sub_centres[1][i]])

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
    u0 = np.zeros([1,F.sub_centres.shape[1]])
    v0 = np.zeros([1,F.sub_centres.shape[1]])

    sub_size = F.sub_size
    for i in range(0,F.sub_centres.shape[1]):
        #coordinates of all subset centres
        centerx = F.sub_centres[0,i]
        centery = F.sub_centres[1,i]
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

#----------------------------------------------------------------------------------------

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
    

def CorrelateImages(image_set, settings):

    pass