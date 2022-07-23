#     IMPORT MODULES
# ------------------------------------------------------------------------

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

#     DEFINE CLASSES AND FUNCTIONS
# ------------------------------------------------------------------------
#general image for DIC analysis
class DICImage:
    def __init__(self,image,setup):
        self.image = image
        self.sub_size =  int(setup[0])
        self.SF_order = setup[1]
        self.GF_stddev = setup[2]
        self.GF_filtsize = int(setup[3])
        self.corr_refstrat = setup[4]
        self.frequency = int(setup[5])
        self.img_rows = setup[6]
        self.img_columns = setup[7]
        #create coordinates of pixels in subset relative to centre, dX dY
        #(constant subset size-these relative coordinates are the same for all subsets)
        #create coordinates for subset centers
        self.x, self.y, self.sub_centres = CreateSubsets(setup)
        #shape function parameters of F, 12 parameters for each subset
        self.P = np.zeros([6, self.sub_centres.shape[1]]) 
        self.sub_halfwidth = (self.sub_size-1)/2         
#reference image F
class ReferenceImage(DICImage):
    def __init__(self,image,setup):
        super().__init__(image,setup)
        #gaussian filter applied to reference image
        #skip blur for now
        self.image = image#cv.GaussianBlur(image,(self.GF_filtsize,self.GF_filtsize),self.GF_stddev)
        #gradient of F in xy directions
        self.F_grady = np.zeros([image.shape[0], image.shape[1]])#np.array(np.gradient(self.image),dtype=float)
        self.F_gradx = np.zeros([image.shape[0], image.shape[1]])
#deformed image G
class DeformedImage(DICImage):
    def __init__(self,image,setup):
        super().__init__(image,setup)
        self.image = image#cv.GaussianBlur(image,(self.GF_filtsize,self.GF_filtsize),self.GF_stddev)
        #interpolation of image at each pixel location, bicubic b-spline
        #interpolation of deformed image
        self.G_interp_coeff = scipy.interpolate.RegularGridInterpolator((np.linspace(0,self.image.shape[0]-1,self.image.shape[0]),np.linspace(0,self.image.shape[1]-1,self.image.shape[1])), self.image)
        self.corr_coeff = np.zeros([1,self.sub_centres.shape[1]])
        self.stop_val = np.zeros([1,self.sub_centres.shape[1]])
        self.iterations = np.zeros([1,self.sub_centres.shape[1]])
#store DIC results
class StoreDICResults:
    def __init__(self, displacement):
        self.displacement = displacement
#plot DIC results
class PlotDICResults(StoreDICResults):
    def __init__(self, displacement):
        super().__init__(displacement)
        #self.meshgrid = 
# create subset center positions
# #
def CreateSubsets(setup):
    sub_size = int(setup[0])
    sub_frequency = int(setup[5])
    img_rows = int(setup[6]) 
    img_columns = int(setup[7])
    #store subset centre coordinates as MESHGRIDS
    sub_centres_y, sub_centres_x = np.meshgrid(np.arange(int(0.5*(sub_size-1)+sub_frequency),
                                                         int(img_rows-0.5*(sub_size-1)),
                                                         sub_frequency),
                                               np.arange(int(0.5*(sub_size-1)+sub_frequency),
                                                         int(img_columns-0.5*(sub_size-1)),
                                                         sub_frequency),
                                               indexing = 'ij')
    #store subset centre coordinates as vectors
    sub_centres_x = np.array([sub_centres_x.flatten(order = 'F')]).T
    sub_centres_y = np.array([sub_centres_y.flatten(order = 'F')]).T
    sub_centers = np.vstack((sub_centres_x.T,sub_centres_y.T))
    #create subset relative coordinates
    [y, x]=np.meshgrid(np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                         np.linspace(-0.5*(sub_size-1),0.5*(sub_size-1),sub_size),
                         indexing='ij')
    #save subsets as column vectors
    x = np.array([x.flatten(order = 'F')]).T
    y = np.array([y.flatten(order = 'F')]).T
    return x, y, sub_centers
#define reference subset and its gradients based on center position
# #
def RefSubsetInfo(F,i):
    #inherited from F (is this ordering right?)
    #xy coordinates of i'th subset centre
    centerx, centery = np.array([F.sub_centres[0][i],F.sub_centres[1][i]])
    #extract  subset information
    f = F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
    #extract subset spatial gradients
    dfdy = F.F_grady[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
    dfdx = F.F_gradx[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                    centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
    #average subset intensity, and sum of squared differences
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    f = np.array([f.flatten(order = 'F')]).T
    return f, f_mean, f_tilde, dfdx, dfdy
#Hessian, used for first order taylor series ezpansion of the objective function
# #
def Hessian(fx,fy,F):
    #store intensity gradients of subset as vectors
    fx = np.array([fx.flatten(order = 'F')]).T
    fy = np.array([fy.flatten(order = 'F')]).T
    #assign subset coordinates
    x = F.x
    y = F.y
    #procuct of f subset gradient and warp function gradient
    fgrad_X_dWdP = np.array([fx[:,0], fx[:,0]*x[:,0], fx[:,0]*y[:,0],fy[:,0], fy[:,0]*x[:,0], fy[:,0]*y[:,0]]).T
    Hess = np.dot(fgrad_X_dWdP.T, fgrad_X_dWdP)   
    return Hess, fgrad_X_dWdP
#Affine transformation warp function
# #
def AffineTrans(G,i):
    #subset deformation parameters/SFP's
    P = G.P[:,i]
    #assign subset coordinates
    x = G.x
    y = G.y
    #displace, stretch and shear subset in xy-coordinates (14):
    #order of SFP's P[i]: 0,  1,  2,  3,  4,  5
    #                     u   ux  uy  v   vx  vy
    deformed_x = (1+P[1])*x + P[2]*y + P[0]
    deformed_y = P[4]*x + (1+P[5])*y + P[3]
    #affine transformation in this case
    deformed_subset = np.hstack([deformed_x, deformed_y])
    return deformed_subset
#define deformed image subsets based on center position
# #
def DefSubsetInfo(G, deformed_subset,i):
    #deform subset to obtain new intensity values(G_interp_coeff is a function of the xy coordinates)
    N_points = deformed_subset.shape[0]
    centerx, centery = np.array([G.sub_centres[0][i],G.sub_centres[1][i]])
    y = centery*np.ones(N_points) + deformed_subset[:,1]
    x = centerx*np.ones(N_points) + deformed_subset[:,0]
    g = np.zeros([N_points,1])
    for m in range(0,N_points):
        g[m] = G.G_interp_coeff(np.array([y[m],x[m]]))
    g_mean = g.mean()
    g_tilde = np.sqrt(np.sum((g[:]-g_mean)**2))
    return g, g_mean, g_tilde
#update SFP's for next iteration of subset correlation
# #
def UpdateSFP(P, deltaP):
    #(22)
    #w of current estimate of SFPs
    #order of SFP's P[i]: 0,  1,  2,  3,  4,  5
    #                     u   ux  uy  v   vx  vy
    w_P = np.array([[1+P[1], P[2], P[0]],
                    [P[4], 1+P[5], P[3]],
                    [0,      0,      1]])
    #w of current deltaP               
    w_dP = np.array([[1+deltaP[1,0], deltaP[2,0], deltaP[0,0]],
                    [deltaP[4,0], 1+deltaP[5,0], deltaP[3,0]],
                    [0,      0,      1]])
    up = np.linalg.solve(w_dP,w_P)
    #(21)
    Pupdate = np.array([[up[0,2]],[up[0,0]-1],[up[0,1]],[up[1,2]],[up[1,0]],[up[1,1]-1]])
    return Pupdate
#convergence criteria for subset correlation (is this actually correct ?)
# #
def StopCriteria(dP, zeta):
    # (23)
    b = np.array([[1,zeta,zeta,1,zeta,zeta]])
    convergence_parameter = np.sqrt(np.sum((dP*b)**2))
    return convergence_parameter
#initial estimate for correlation using Fourier analysis
# #
def EstimateDisplacementsFourier(F,G):
    u0 = np.zeros([1,F.sub_centres.shape[1]])
    v0 = np.zeros([1,F.sub_centres.shape[1]])
    sub_size = F.sub_size
    for i in range(0,F.sub_centres.shape[1]):
        centerx, centery = np.array([F.sub_centres[0][i],F.sub_centres[1][i]])
    #extract  subset information
        f = F.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
        g = G.image[centery-int(0.5*(F.sub_size-1)):centery+int(0.5*(F.sub_size-1))+1,
                centerx-int(0.5*(F.sub_size-1)):centerx+int(0.5*(F.sub_size-1))+1]
        #normalised cross power spectrum
        cross_power_spec = np.fft.fft2(f)*np.fft.fft2(g).conj()/abs(np.fft.fft2(f)*np.fft.fft2(g).conj())
        #frequency domain correlation coefficients
        corr_coeff = abs(np.fft.ifft2(cross_power_spec))
        max_CC = np.where(corr_coeff == np.max(corr_coeff[:]))
        #fourier shift subset coordinates
        index_shift = -1*np.fft.ifftshift(np.array([np.linspace(-np.fix(sub_size/2),
                                                                math.ceil(sub_size/2)-1,
                                                                num=sub_size)]))
        #estimate of initial displacements
        u0[0,i] = index_shift[0,max_CC[1]]
        v0[0,i] = index_shift[0,max_CC[0]]
    return u0, v0

    #initial estimate for correlation using ORB
# #
# def EstimateDisplacementsORB(F, G, i, sub_size):
#     #coordinates of all subset centres
#     centerx = F.sub_centres[0,i]
#     centery = F.sub_centres[1,i]
#     #extract subset based on index i
#     f = F.image[centery-int(0.5*(sub_size-1)):centery+int(0.5*(sub_size-1))+1,
#                 centerx-int(0.5*(sub_size-1)):centerx+int(0.5*(sub_size-1))+1]

#     g = G.image[centery-int(0.5*(sub_size-1)):centery+int(0.5*(sub_size-1))+1,
#                 centerx-int(0.5*(sub_size-1)):centerx+int(0.5*(sub_size-1))+1]
#     #create ORB object
#     descriptor_extractor = ORB(downscale=1.1,n_keypoints=20,fast_n=5,fast_threshold=0.15)
#     #f keypoints and descriptors
#     descriptor_extractor.detect_and_extract(f)
#     keypoints1 = descriptor_extractor.keypoints
#     descriptors1 = descriptor_extractor.descriptors
#     #g keypoints and descriptors
#     descriptor_extractor.detect_and_extract(g)
#     keypoints2 = descriptor_extractor.keypoints
#     descriptors2 = descriptor_extractor.descriptors
#     #image matching
#     matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
#     #create input data for least-squares
#     n_matches = len(matches)
#     X = np.zeros((2* n_matches, 4))
#     y = np.zeros((2* n_matches, 1))
#     for i in range(0, 2* n_matches, 2):
#         y[i]   = keypoints2[matches[int(i/2)][1]][0]
#         y[i+1] = keypoints2[matches[int(i/2)][1]][1]
#         locX = keypoints1[matches[int(i/2)][0]][0]
#         locY = keypoints1[matches[int(i/2)][0]][1]
#         X[i][0] = 1.0
#         X[i][1] = locX
#         X[i+1][2] = 1.0
#         X[i+1][3] = locY    
#     #initialize least-squares
#     resRLM = None
#     # A robust fit of the model/iteratively reweighted least squares
#     modelRLM = sm.RLM( y, X )
#     resRLM = modelRLM.fit(maxiter = 20, tol = 1e-4)
#     #determine diplcacements from calculated coefficients
#     coefficients = resRLM.params
#     u = -coefficients[0]
#     v = -coefficients[3]

#     return u,v,-coefficients