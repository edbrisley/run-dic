#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import dic
import math as m
%matplotlib inline

#RANSAC/SKIMAGE packages
from skimage import data
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform, PolynomialTransform, ProjectiveTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac
import copy
from numpy import loadtxt
import numpy.linalg as la


#%%
#load images and and settings

#load image set from directory
image_set = dic.LoadImages()
#load settings from text file
settings = dic.LoadSettings(image_set)
#no of images in directory
N_images = len(image_set)
print(image_set)
print(settings)


#%%
#initialise reference image objects from both cameras for subset matching
#load reference image of first camera
F1 = dic.ReferenceImage(cv.imread('images2/R0.tif', 0), settings)
#specify ROI in reference image
ROI = cv.selectROI("Specify measurement area:", F1.image)
#store ROI intensity values as array
F1_ROI = F1.image_8bit[int(ROI[1]):int(ROI[1]+ROI[3]),
                  int(ROI[0]):int(ROI[0]+ROI[2])]
#create subsets and initialise SFPs
F1.CreateSubsets(settings, ROI)
#similarly for F2
F2 = dic.DeformedImage(cv.imread('images2/L0.tif', 0), settings)
F2.InitCorrelationParams(F1)


#%%
#create padded image for ROI, keypoint matching
F1_ROI_padded = np.zeros([F1.image.shape[0], F1.image.shape[1]])
F1_ROI_padded[int(ROI[1]):int(ROI[1]+ROI[3]),
              int(ROI[0]):int(ROI[0]+ROI[2])] = F1_ROI
F1_ROI_padded = F1_ROI_padded.astype(np.uint8)


#%%
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

print('ROI in camera 1:')
plt.imshow(F1_ROI_padded, cmap = 'gray')
plt.show()

print('Entire FOV from camera 2:')
plt.imshow(F2.image, cmap = 'gray')
plt.show()


#%%
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


#%%
#identify K nearest keypoints to each subset centre

k = 10
kNN_indices = np.zeros([N_subsets,k])

for i in range(0, N_subsets):
    #SSD between subset centre i and all keypoints
    diff = np.sqrt((Xo1[0,i]-Xk1[0,:])**2 + (Xo1[1,i]-Xk1[1,:])**2)
    idx = np.argpartition(diff, k)
    #k minimum SSD between subset centre i and all keypoints
    kNN_indices[i, :] = idx[:k]

print('indices:', idx[:k])
print('\nSSD:', diff[idx[:k]])
print('\nkeypoint indices:',kNN_indices[0,:])


#%%
#display RANSAC results for signle subset

#keypoint coordinates corresponding to subset
xkp1 = Xk1[:,kNN_indices[350,:].astype(int)].T
xkp2 = Xk2[:,kNN_indices[350,:].astype(int)].T

print(xkp1)
print(xkp2)

# robustly estimate affine transform model with RANSAC
#perform RANSAC FOR j'th SUBSET
model_robust, inliers = ransac((xkp1, xkp2), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)

#results
print('Transformation model parameters:\n', model_robust.params[0], '\n',model_robust.params[1] )
print('Inliers logical matrix:\n', inliers)
#
print("RANSAC:")
print(f'Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), '
      f'Translation: ({model_robust.translation[0]:.4f}, '
      f'{model_robust.translation[1]:.4f}), '
      f'Rotation: {model_robust.rotation:.4f}')


#%%
#display RANSAC results for signle subset
j = 350
#keypoint coordinates corresponding to subset
xkp1 = Xk1[:,kNN_indices[j,:].astype(int)].T
xkp2 = Xk2[:,kNN_indices[j,:].astype(int)].T

xo1 = F1.sub_centers[0, j]
yo1 = F1.sub_centers[1, j]
# robustly estimate affine transform model with RANSAC
#perform RANSAC FOR j'th SUBSET
model_robust, inliers = ransac((xkp1, xkp2), ProjectiveTransform, min_samples=4,
                               residual_threshold=2, max_trials=100)
#results
print('\nTransformation model parameters:\n', model_robust.params)
print('\nInliers logical matrix:\n', inliers)
#


#%%
#loop through all subsets, fit affine model between left and right images using RANSAC


for j in range(0, N_subsets):
    #fetch centre coordinates of current subset in F1 (reference image 1)
    xo1 = F1.sub_centers[0, j]
    yo1 = F1.sub_centers[1, j]

    #find kNN keypoint coordinates for current subset
    xkp1 = Xk1[:,kNN_indices[j,:].astype(int)].T
    xkp2 = Xk2[:,kNN_indices[j,:].astype(int)].T
    model_robust, inliers = ransac((xkp1, xkp2), ProjectiveTransform, min_samples=3,
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
    F2.sub_centers[:, j] = xo2, yo2
