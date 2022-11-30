import numpy as np
from fast_interp import interp2d
import cv2 as cv
import scipy as sp
from numpy import copy
from fast_interp import interp2d
import scipy.sparse.linalg as splalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#
def FastInterpolation(image):

    #image coordinates
    ny = image.shape[0]
    nx = image.shape[1]

        #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    return image_interpolated

def Q4SFMatrix():
    #node coordinates in psi_eta coordinate system
    psi_eta = (1/np.sqrt(3))*np.array([[-1, -1],
                                        [1, -1],
                                        [1,  1],
                                        [-1,  1]])
    #initialise shape function matrix, 4 shape functions for each node in element
    N = np.zeros([4,4])
    #shape functions for each node in element
    N1 = (1/4)*(1-psi_eta[:, 0])*(1-psi_eta[:,1])
    N2 = (1/4)*(1+psi_eta[:, 0])*(1-psi_eta[:,1])
    N3 = (1/4)*(1+psi_eta[:, 0])*(1+psi_eta[:,1])
    N4 = (1/4)*(1-psi_eta[:, 0])*(1+psi_eta[:,1])
    #assemble shape functions into element shape function matrix
    N[:, 0] = N1
    N[:, 1] = N2
    N[:, 2] = N3
    N[:, 3] = N4

    #initialise shape function derivative matrices
    dNdxsi = np.zeros([4,4])
    dNdeta = np.zeros([4,4])
    #shape function derivatives
    dN1dxsi = (1/4)*(psi_eta[:, 1] - 1)
    dN2dxsi = (1/4)*(-psi_eta[:, 1] + 1)
    dN3dxsi = (1/4)*(psi_eta[:, 1] + 1)
    dN4dxsi = (1/4)*(-psi_eta[:, 1] - 1)
    #
    dN1deta = (1/4)*(psi_eta[:, 0] - 1)
    dN2deta  = (1/4)*(-psi_eta[:, 0] - 1)
    dN3deta  = (1/4)*(psi_eta[:, 0] + 1)
    dN4deta  = (1/4)*(-psi_eta[:, 0] + 1)

    dNdxsi[:, 0] = dN1dxsi
    dNdxsi[:, 1] = dN2dxsi
    dNdxsi[:, 2] = dN3dxsi
    dNdxsi[:, 3] = dN4dxsi

    dNdeta[:, 0] = dN1deta
    dNdeta[:, 1] = dN2deta
    dNdeta[:, 2] = dN3deta
    dNdeta[:, 3] = dN4deta


    return N, dNdxsi, dNdeta

def MeshS14(freq):

    sub_centers_y, sub_centers_x = np.meshgrid(np.arange(25,
                                                    570,
                                                    freq),
                                           np.arange(25,
                                                    2030,
                                                    freq),
                                                    indexing = 'ij')

    #number of x and y coords
    N_coords_y = sub_centers_y.shape[0]
    N_coords_x = sub_centers_y.shape[1]

    #vectors
    node_coords = np.vstack((np.array([sub_centers_x.flatten(order = 'F')]), np.array([sub_centers_y.flatten(order = 'F')])))
    N_nodes = node_coords.shape[1]
    #nodes xy coordinates

    node_coords= node_coords.T

    #create element connectivity table (nodes of each element)
    mesh_node_no = np.arange(0, N_nodes).reshape(N_coords_y, N_coords_x, order = 'F')
    N_elements = (N_coords_x-1)*(N_coords_y-1)
    element_conn = np.zeros([N_elements, 4]).astype(int)
    l = 0
    for j in range(0, N_coords_x - 1):
        #rows
        for i in range(0, N_coords_y - 1):
            element_conn[l, :] = np.array([mesh_node_no[i,j], mesh_node_no[i+1, j], mesh_node_no[i+1, j+1], mesh_node_no[i, j+1]])
            l = l + 1
    

    #CONNECTIVITY (in ICGN script)
    dof = np.arange(N_nodes)
    dof = np.c_[dof, dof + N_nodes*(dof>=0)]
    N_dof = 2*N_nodes
    N_ip = 4*N_elements

    return node_coords, element_conn, dof, N_dof, N_elements, N_ip

def FastInterpolation(image):
    
    #image coordinates
    ny = image.shape[0]
    nx = image.shape[1]
    #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    
    return image_interpolated

def Tikhonov(dphixdx, dphixdy, dphiydx, dphiydy, wdetJ):

    wdetJ = sp.sparse.diags(wdetJ)
    L = (dphixdx.T@wdetJ@dphixdx +
        dphiydy.T@wdetJ@dphiydy +
        dphixdy.T@wdetJ@dphixdy +
        dphiydx.T@wdetJ@dphiydx)

    return L

def Hessian(N_globalX_x_coords, N_globalY_x_coords, F_interp, N_global_x, N_global_y, wdetJ):

    Xs = N_globalX_x_coords
    Ys = N_globalY_x_coords

    f = np.array([F_interp(Ys, Xs)]).T

    dfdx = dFdX_interp(Ys, Xs)
    dfdy = dFdY_interp(Ys, Xs)

    delF_dphidq = (
                    sp.sparse.diags(dfdx + dfdy)@N_global_x
                  + sp.sparse.diags(dfdx + dfdy)@N_global_y )
    w_delF_dphidq = sp.sparse.diags(wdetJ)@delF_dphidq
    
    f_range = np.max(f) - np.min(f)
    f_ave = np.mean(f)
    f_std = np.std(f)
    f = f - f_ave
    Hess = delF_dphidq.T@w_delF_dphidq

    return [Hess, f, f_std, f_ave, f_range, w_delF_dphidq]

def Residual(Hess, G_interp, N_globalX_x_coords, N_global_x, N_globalY_x_coords, N_global_y, U):
    
    x, y = N_globalX_x_coords + N_global_x@U, N_globalY_x_coords + N_global_y@U
    f = Hess[1]
    f_std = Hess[2]
    w_delF_dphidq = Hess[5]

    res = np.array([G_interp(y, x)]).T
    res = res - np.mean(res)
    res_std = np.std(res)
    res = f - f_std/res_std*res

    B = w_delF_dphidq.T@res

    return B[:, 0], res

def RegularizeHT(H, L, node_coords, l0, dof, N_dof):
    
    used_nodes = dof[:, 0] > 0
    V = np.zeros(N_dof)
    V[dof[used_nodes, 0]] = np.cos(node_coords[used_nodes, 1]/l0*2*np.pi)
    H0 = V.dot(H.dot(V))
    L0 = V.dot(L.dot(V))
    l = H0/L0

    return H , l , L#H + l * L

#--------- TEST CASE: SAMPLE 14
#images
#F = cv.GaussianBlur(cv.imread('images/img1.tif', 0).astype('double'), (5,5), 0.5)
F = cv.GaussianBlur(cv.normalize(cv.imread('images/img0.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)
F_interp = FastInterpolation(F)
deltaF = np.array(np.gradient(F),dtype= float)
dFdY = deltaF[0]
dFdX = deltaF[1]
dFdX_interp = FastInterpolation(dFdX)
dFdY_interp = FastInterpolation(dFdY)


#G = cv.GaussianBlur(cv.imread('images/img1.tif', 0).astype('double'), (5,5), 0.5)
G = cv.GaussianBlur(cv.normalize(cv.imread('images/img1.tif', 0).astype('double'),
                                  None, 0.0, 1.0, cv.NORM_MINMAX),
                                  (5,5), 0.5)
G_interp = FastInterpolation(G)

#numerical integration
##____________________________________________________________________________
#mesh parameters
Ni_matrix, dNi_xi_matrix, dNi_eta_matrix  = Q4SFMatrix()
wg = np.ones(4)
freq = 10
l0 = 200
node_coords, element_conn, dof, N_dof, N_elements, N_ip = MeshS14(freq)
#xy-coordinates connectivity table
xn = node_coords[element_conn, 0]
yn = node_coords[element_conn, 1]
N_nodes = np.unique(element_conn).shape[0]
#initialize global arrays
wdetJ = np.array([])
col = np.array([], dtype=int)
row = np.array([], dtype=int)
N_global = np.array([])
dphidx_global = np.array([])
dphidy_global = np.array([])

x_dofs_mesh = dof[element_conn, 0]
xn = node_coords[element_conn, 0]
yn = node_coords[element_conn, 1]

N_sf = Ni_matrix.shape[1]

wdetJj = np.zeros(N_ip)
rowj = np.zeros(N_ip*N_sf, dtype=int)
colj = np.zeros(N_ip*N_sf, dtype=int)
Nj = np.zeros(N_ip*N_sf)
dphidx_j = np.zeros(N_ip*N_sf)
dphidy_j = np.zeros(N_ip*N_sf)

index0 = 0
for el_i in range(N_elements):
    #integration points
    
    indices_el = index0 + np.arange(4)
    J11 = dNi_xi_matrix@xn[el_i, :]
    J12 = dNi_xi_matrix@yn[el_i, :]
    J21 = dNi_eta_matrix@xn[el_i, :]
    J22 = dNi_eta_matrix@yn[el_i, :]
    detJ = J11*J22 - J12*J21
    #
    wdetJj[indices_el] = wg*abs(detJ)
    [el_cols, el_rows] = np.meshgrid(x_dofs_mesh[el_i, :], indices_el + len(wdetJ))
    ind = N_sf*index0 + np.arange(np.prod(Ni_matrix.shape))
    rowj[ind] = el_rows.ravel()
    colj[ind] = el_cols.ravel()
    Nj[ind] = Ni_matrix.ravel()
    #
    
    dphidx = (J22/detJ)[:, np.newaxis]*dNi_xi_matrix + (-J12/detJ)[:,np.newaxis]*dNi_eta_matrix
    dphidy = (-J21/detJ)[:, np.newaxis]*dNi_xi_matrix + (J11/detJ)[:, np.newaxis]*dNi_eta_matrix
    dphidx_j[ind] = dphidx.ravel()
    dphidy_j[ind] = dphidy.ravel()
    index0 += 4
#
index_f = index0
col = np.append(col, colj[:index_f*N_sf])
row = np.append(row, rowj[:index_f*N_sf])
N_global = np.append(N_global, Nj[:index_f*N_sf])

dphidx_global = np.append(dphidx_global, dphidx_j[:index_f*N_sf])
dphidy_global = np.append(dphidy_global, dphidy_j[:index_f*N_sf])
#
wdetJ = np.append(wdetJ, wdetJj[:index_f])    
N_ip = len(wdetJ)

N_global_x = sp.sparse.csc_matrix(
                                    (N_global, (row, col)),
                                    shape=(N_ip, N_dof))
N_global_y = sp.sparse.csc_matrix(
                                    (N_global, (row, col + N_dof//2)),
                                    shape=(N_ip, N_dof))

dphixdx = sp.sparse.csc_matrix(
                                (dphidx_global, (row, col)),
                                shape=(N_ip, N_dof))
dphixdy = sp.sparse.csc_matrix(
                                (dphidy_global, (row, col)),
                                shape=(N_ip, N_dof))
dphiydx = sp.sparse.csc_matrix(
                                (dphidx_global,(row, col + N_dof//2)),
                                shape=(N_ip, N_dof))
dphiydy = sp.sparse.csc_matrix(
                                (dphidy_global, (row, col + N_dof//2)),
                                shape=(N_ip, N_dof))
        #
coords_vector = np.zeros(N_dof)
(rep,) = np.where(dof[:, 0] >= 0)
coords_vector[dof[rep, :]] = node_coords[rep, :]

N_globalX_x_coords = N_global_x.dot(coords_vector)
N_globalY_x_coords = N_global_y.dot(coords_vector)
##____________________________________________________________________________


#intial displacements
U0 = np.zeros(N_dof)
U = copy(U0)

#Hessian
Hess = Hessian(N_globalX_x_coords, N_globalY_x_coords, F_interp, N_global_x, N_global_y, wdetJ)
H = Hess[0]
#regularization
L = Tikhonov(dphixdx, dphixdy, dphiydx, dphiydy, wdetJ)

H, l, L = RegularizeHT(H, L, node_coords, l0, dof, N_dof)
H_reg = H + l*L
print('\nfreq:',freq,'\nl0',l0)
##____________________________________________________________________________
print('Check symmetry:\n', np.max(abs(H -  H.T)), np.min(abs(H -  H.T)))
print('Hessian',  H.shape)

# cond  = np.linalg.cond(H.toarray())
# det = np.linalg.det(H.toarray())
# print('before regularization')
# print('cond no.:',  cond)
# print('determinant:', det)
#print('SPD', np.all(np.linalg.eigvals(H)>0))

# cond  = np.linalg.cond(H_reg.toarray())
# det = np.linalg.det(H_reg.toarray())
# print('\nafter regularization')
# print('cond no.:',  cond)
# print('determinant:', det)
#print('SPD', np.all(np.linalg.eigvals(H_reg.toarray())>0))
##____________________________________________________________________________

#H_LU = splalg.splu(H_reg)
H_LU = splalg.splu(H_reg)
#index ranges for xy displacements
x_dofs = np.arange(N_dof//2)
y_dofs = np.arange(N_dof//2, N_dof)
#U[dof[:, 0]
phi = N_global_x[:, x_dofs]
MM = np.dot(phi.T, phi)
MMLU = splalg.splu(MM.T)
dNdx = MMLU.solve(phi.T.dot(dphixdx[:, x_dofs]).toarray())
dNdy = MMLU.solve(phi.T.dot(dphixdy[:, x_dofs]).toarray())

#ICGN iterations
for k in range(0,60):
    b, res = Residual(Hess, G_interp, N_globalX_x_coords, N_global_x, N_globalY_x_coords, N_global_y, U)
    # print('H_LU', H_LU.shape)
    # print('b', b.shape)
    
    dU = H_LU.solve(b - l*L.dot(U))#dU = H_LU.solve(b) #
    # ICGN Correction:
    # print('dNdx', dNdx.shape)
    # print('dNdy', dNdy.shape)
    # print('dU[x_dofs]', dU[x_dofs].shape)
    # print('U[x_dofs]', U[x_dofs].shape)
    Ux = dNdx.dot(U[x_dofs])*dU[x_dofs] + dNdy.dot(U[x_dofs])*dU[y_dofs]
    Uy = dNdx.dot(U[y_dofs])*dU[x_dofs] + dNdy.dot(U[y_dofs])*dU[y_dofs]
    U = U + dU + np.append(Ux, Uy)
    #U += dU
    if k ==0:
        print('dU', dU.shape)
        # print('Ux', Ux.shape)
        # print('Uy', Uy.shape)
    err = np.linalg.norm(dU)/np.linalg.norm(U)
    print("Iter # %2d | disc/dyn=%2.2f gl | dU/U=%1.2e" % (k + 1, np.std(res), err))
    if err<1e-3:
        break


XY = np.zeros([N_nodes, 2])
for i in range(0, N_nodes):    
    XY[i, :] = (node_coords[:, 1][i], node_coords[:, 0][i])

grid_x, grid_y = np.mgrid[0:2048:1, 0:589:1]
#U[0:int(N_dof//2)]
U = -U
U_new_grid = griddata(XY, U[dof[:, 0]], (grid_y, grid_x), method='cubic')
U_new_grid = U_new_grid.T

plt.imshow(U_new_grid, extent=(0,2048,0,589),cmap = 'jet', aspect = '0.8')
plt.colorbar(location = 'bottom', shrink = 0.4)
plt.xlabel('X')
plt.ylabel('Y')
#plt.title('X-displacement/U, Global solver\n\nSample14 L5 Amp0.1\n element size = {}'.format(freq))
print('displ. extremes', np.min(U[dof[:, 0]]), np.max(U[dof[:, 0]]))
print('double_amp', np.max(U[dof[:, 0]]) - np.min(U[dof[:, 0]]))
plt.show()




print('return(0)')