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
#reference image
F = cv.GaussianBlur(cv.normalize(cv.imread('images/ref.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)
#deformed image
G = cv.GaussianBlur(cv.normalize(cv.imread('images/def.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)

#reference image data, processing, interpolation
F_interp = gd.FastInterpolation(F)
dFdY, dFdX = np.array(np.gradient(F), dtype=float)
dFdX_interp = gd.FastInterpolation(dFdX)
dFdY_interp = gd.FastInterpolation(dFdY)

#
G_interp = gd.FastInterpolation(G)

data = dict()
data['F_interp'] = F_interp
data['dFdY'] = dFdY
data['dFdX'] = dFdX
data['dFdX_interp'] = dFdX_interp
data['dFdY_interp'] = dFdY_interp
data['G_interp'] = G_interp

#Gaussian quadrature rule for numerical integration
#external call to RosettaCode library to return the
#interpolation point coordinates and the quadrature weights
q_rule = 11
xsi_eta, w2 = gd.GaussQuadrature2D(q_rule)

#shape function matrix for a single element
N  = gd.Q8SFMatrix(xsi_eta, w2)
n_sf = N[0].shape[1]
n_q = N[0].shape[0]
wg = N[3]

#global-DIC user specified settings ()
#element size
el_size = 21
#regularization length
alpha2 = 0.01
#image region-of-interest coordinates
ROIcoords = np.array([5, 3995, 50, 460])


#create the FE element mesh
node_coords, element_conn, dof, n_dof, n_elements, n_ip, n_nodes = gd.Q8RectangularMesh(el_size, ROIcoords, n_q)
#xy-coordinates connectivity table
xn = node_coords[element_conn, 0]
yn = node_coords[element_conn, 1]
x_dofs_mesh = dof[element_conn, 0]

#assign variables to mesh dictionary
mesh = dict()

#user-defined settings
mesh['el_size'] = el_size
mesh['alpha2'] = alpha2
mesh['ROIcoords'] = ROIcoords

#details of mesh assembly
mesh['node_coords'] = node_coords
mesh['element_conn'] = element_conn
mesh['dof'] = dof
mesh['n_dof'] = n_dof
mesh['n_nodes'] = n_nodes
mesh['n_elements'] = n_elements
mesh['n_ip'] = n_ip
mesh['xn'] = xn
mesh['yn'] = yn
mesh['x_dofs_mesh'] = x_dofs_mesh
mesh['N'] = N
mesh['wg'] = wg

gd.GlobalAssembly(mesh)

#initialise displacements to [0 , ... , 0] for the DIC challenge
d0 = np.zeros(n_dof)

#compute the nodal displacements using global DIC
d = gd.RegularizedModifiedGN(mesh, data, d0)

XY = np.zeros([n_nodes, 2])
for i in range(0, n_nodes):    
    XY[i, :] = (node_coords[:, 1][i], node_coords[:, 0][i])

grid_x, grid_y = np.mgrid[0:3999:1, 0:500:1]

d_new_grid = griddata(XY, d[dof[:, 1]], (grid_y, grid_x), method='cubic')
d_new_grid = d_new_grid.T

print('displ. extremes', np.min(d[dof[:, 1]]), np.max(d[dof[:, 1]]))
plt.imshow(d_new_grid, extent=(0,3999,0,500),cmap = 'jet', aspect = '0.8')
plt.colorbar(location = 'bottom', shrink = 0.4)
plt.title('V-displacement\nQ8 element size = {}x{} pixels\nTikhonov regularization: alpha = {}\n{}x{} point GQ rule'.format(el_size, el_size, alpha2, q_rule, q_rule))
plt.clim(-0.5,0.5)
plt.xlabel('X (pix)')
plt.ylabel('Y (pix)')
plt.show()