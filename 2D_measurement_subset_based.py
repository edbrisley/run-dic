#import some libraries
import dic
import cv2 as cv
import numpy as np
import scipy.interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#%%
##load image set from directory
image_set = dic.LoadImages()
#load settings from config file
settings = dic.LoadSettings(image_set)
#no of images in directory
N_images = len(image_set)
#%%
#initialise reference image object
F = dic.ReferenceImage(cv.imread("images/{}".format(image_set[0]),0), settings)
#create subsets and initialise SFPs
F.CreateSubsets(settings)
G = dic.DeformedImage(cv.imread("images/{}".format(image_set[1]),0), settings)
G.InitCorrelationParams(F)
#%%
#initial estimation of deformed image subset centre coordinates (in the coordinate system of the reference image)
#check SF order:
if settings['ShapeFunctionOrder'] == 1:
    G.sub_centers[0,:] = F.sub_centers[0,:] + np.round(F.P[0,:])
    G.sub_centers[1,:] = F.sub_centers[1,:] + np.round(F.P[3,:])
    #initial estimate of the shape function as rigid body translation
    G.P[0,:], G.P[3,:] = dic.EstimateDisplacementsFourier(F, G)
else:
    G.sub_centers[0,:] = F.sub_centers[0,:] + np.round(F.P[0,:])
    G.sub_centers[1,:] = F.sub_centers[1,:] + np.round(F.P[6,:])
    #initial estimate of the shape function as rigid body translation
    G.P[0,:], G.P[6,:] = dic.EstimateDisplacementsFourier(F, G)
#%%
# run 2D subset-based correlation
dic.CorrelateImages2D(F, G, settings)
#%%
#plot x-displacement results
u = G.P[0, :]
print('\netremes', np.max(u), np.min(u), '\ndouble_amp:', np.max(u) - np.min(u))
N_subsets = G.P.shape[1]
N_rows = G.image.shape[0]
N_cols = G.image.shape[1]
yo = G.sub_centers[1, :]
xo = G.sub_centers[0, :]
#
XY = np.zeros([N_subsets, 2])
for i in range(0, N_subsets):
    XY[i, :] = (yo[i], xo[i])
grid_x, grid_y = np.mgrid[0:N_cols:1, 0:N_rows:1]
U_new_grid = griddata(XY, u, (grid_y, grid_x), method='cubic')
U_new_grid = U_new_grid.T
#
plt.imshow(U_new_grid, extent=(0,N_cols ,0, N_rows),cmap = 'jet', aspect = '0.8')
plt.colorbar(location = 'bottom', shrink = 0.4)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X-displacement/U, Affine transformation model\n\nSample14 L5 Amp0.1\nsubset size = {}, subset frequency = {}'.format(G.sub_size, G.frequency))
plt.show()
