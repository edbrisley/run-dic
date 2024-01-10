import numpy as np
import cv2 as cv
import scipy as sp
from numpy import copy
import scipy.sparse.linalg as splalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from operator import itemgetter
import GDIC_library as gd

#DIC images: star5 of the 2D-DIC challenge 2.0
#pre-blur the intensity data
#reference image
F = cv.GaussianBlur(cv.normalize(cv.imread('images/ref.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)
#deformed image
G = cv.GaussianBlur(cv.normalize(cv.imread('images/def.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)

data = gd.ProcessImageData(F, G)

#global-DIC user specified settings ()
#Gaussian quadrature rule for numerical integration
q_rule = 11
#element size
el_size = 21
#regularization length
alpha2 = 0.01
#image region-of-interest coordinates
ROIcoords = np.array([5, 3995, 50, 460])
#element type to mesh
el_type = 'Q8'

#create the FE mesh dictionary with settings and mesh details
mesh = gd.FEMesh(q_rule, el_type, el_size, alpha2, ROIcoords)

#initialise displacements to [0 , ... , 0] for the DIC challenge
d0 = np.zeros(mesh['n_dof'])

#compute the nodal displacements using global DIC
d = gd.RegularizedModifiedGN(mesh, data, d0)

#visualize the results
XY = np.zeros([mesh['n_nodes'], 2])
for i in range(0, mesh['n_nodes']):    
    XY[i, :] = (mesh['node_coords'][:, 1][i], mesh['node_coords'][:, 0][i])

grid_x, grid_y = np.mgrid[0:3999:1, 0:500:1]

v_nodal_new_grid = griddata(XY, d[mesh['dof'][:, 1]], (grid_y, grid_x), method='cubic')
v_nodal_new_grid = v_nodal_new_grid.T

plt.imshow(v_nodal_new_grid, extent=(0,3999,0,500),cmap = 'jet', aspect = '0.8')
plt.colorbar(location = 'bottom', shrink = 0.4)
plt.title('V-displacement\nQ8 element size = {}x{} pixels\nTikhonov regularization: alpha = {}\n{}x{} point GQ rule'.format(el_size, el_size, alpha2, q_rule, q_rule))
plt.clim(-0.5,0.5)
plt.xlabel('X (pix)')
plt.ylabel('Y (pix)')
plt.show()
