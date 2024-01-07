import numpy as np
from fast_interp import interp2d
import cv2 as cv
import scipy as sp
from numpy import copy
import scipy.sparse.linalg as splalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from operator import itemgetter
#/
def FastInterpolation(image):

    #image coordinates
    ny = image.shape[0]
    nx = image.shape[1]

        #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    return image_interpolated
#/
def Q8RectangularMesh(el_size, ROIcoords, N_q):
    """Updated Q8 element mesher for rectangular ROI"""
    #-------------------------------------------------------------------------------------
    #ROI boundary coordinates
    x_start, x_end, y_start, y_end = ROIcoords[0], ROIcoords[1], ROIcoords[2], ROIcoords[3]
    #dummy grid to count number of rows and columns in mesh
    dummy_node_grid = np.meshgrid(np.arange(y_start,
                                            y_end,
                                            (el_size-1)/2),
                                    np.arange(x_start,
                                            x_end,
                                            (el_size-1)/2),
                                            indexing = 'ij')
    #count number of rows and columns in node grid
    N_node_rows = dummy_node_grid[0].shape[0]
    N_node_cols = dummy_node_grid[0].shape[1]
    
    #mesh 
    N_elements = int(((N_node_rows -1)/2)*((N_node_cols-1)/2))
    N_nodes = (N_node_cols*N_node_rows - N_elements)
    n_ip = N_q*N_elements #hardcoded for 9PQ
    N_dof = int(2*N_nodes)
    
    #element connectivity table sub-funcion
    #node numbers of first element in mesh
    el1 = np.array([0,
                    2,
                    3+(3*N_node_rows-1)/2,
                    1+(3*N_node_rows-1)/2,
                    1,
                    N_node_rows+1,
                    2+(3*N_node_rows-1)/2,
                    N_node_rows])
    #Q8 element connectivity table
    element_conn = np.zeros([N_elements, 8]).astype(int)
    start = 0
    for col in range(0, int((N_node_cols-1)/2)):
        for row in range(0, int((N_node_rows-1)/2)):
            element_conn[int(start+row), :] = el1 + row*np.array([2, 2, 2, 2, 2, 1, 2, 1])
        #increment element info
        start = start + (N_node_rows-1)/2
        el1 = el1 + (1+(3*N_node_rows-1)/2)
    
    #node coordinates sub-function
    node_coords = np.zeros([N_nodes, 2])
    y_odd = np.arange(y_start, y_end, (el_size-1))
    row_start = 0
    col_start = 0
    for x in range(0, N_node_cols):
        if x%2 == 0:
            node_coords[row_start:row_start + N_node_rows, :] = np.vstack(( np.repeat(x_start + x*(el_size-1)/2, N_node_rows),
                                                                            np.arange(y_start, y_end, (el_size-1)/2))).T
            row_start = row_start + N_node_rows
        else:
            node_coords[row_start:row_start + int((N_node_rows-1)/2 + 1), :] = np.vstack((  np.repeat(x_start + x*(el_size-1)/2,
                                                                                                    (N_node_rows-1)/2 + 1),

                                                                                            y_odd)).T
            row_start = int(row_start + (N_node_rows-1)/2 + 1)
    
    #concatenate x-dofs and y-dofs
    x_dofs = np.array(np.arange(N_nodes))
    y_dofs = np.array(N_nodes + np.arange(N_nodes))
    dof = np.vstack(( x_dofs,
                      y_dofs )).T
    
    mesh = dict()
    mesh['NodeCoordinates'] = node_coords
    mesh['ElementConnectivity'] = element_conn
    mesh['DOFIndices'] = dof
    mesh['n_Nodes'] = N_nodes
    mesh['n_Elements'] = N_elements
    mesh['n_ip'] = n_ip

    #return node_coords, element_conn, dof, N_dof, N_elements, n_ip, N_nodes
    return None
#/
def Q4RectangularMesh(el_size,  ROIcoords, N_q):
    "FE mesh for Q4 element type"

    x_start, x_end, y_start, y_end = ROIcoords[0], ROIcoords[1], ROIcoords[2], ROIcoords[3]
    #dummy grid to count number of rows and columns in mesh
    nodes_y, nodes_x = np.meshgrid(np.arange(y_start,
                                            y_end,
                                            (el_size-1)/2),
                                    np.arange(x_start,
                                            x_end,
                                            (el_size-1)/2),
                                            indexing = 'ij')

    #count the number of rows and columns in the mesh
    N_node_rows = nodes_y.shape[0]
    N_node_cols = nodes_y.shape[1]
    N_elements = (N_node_cols-1)*(N_node_rows-1)

    #vectors
    node_coords = np.vstack((np.array([nodes_x.flatten(order = 'F')]), np.array([nodes_y.flatten(order = 'F')])))
    N_nodes = node_coords.shape[1]
    #nodes xy coordinates
    node_coords= node_coords.T

    #define the node numbers as they appear in the mesh
    mesh_node_no = np.arange(0, N_nodes).reshape(N_node_rows,
                                                N_node_cols,
                                                order = 'F')
    
    #element connectivity table
    element_conn = np.zeros([N_elements, 4]).astype(int)
    l = 0
    for j in range(0, N_node_cols - 1):
        #rows
        for i in range(0, N_node_rows - 1):
            element_conn[l, :] = np.array([mesh_node_no[i,j], mesh_node_no[i+1, j], mesh_node_no[i+1, j+1], mesh_node_no[i, j+1]])
            l = l + 1
    
    #CONNECTIVITY (in ICGN script)
    dof = np.arange(N_nodes)
    dof = np.c_[dof, dof + N_nodes*(dof>=0)]
    N_dof = 2*N_nodes
    n_ip = N_q*N_elements

    mesh = dict()
    mesh['NodeCoordinates'] = node_coords
    mesh['ElementConnectivity'] = element_conn
    mesh['DOFIndices'] = dof
    mesh['n_Nodes'] = N_nodes
    mesh['n_Elements'] = N_elements
    mesh['n_ip'] = n_ip

    #return node_coords, element_conn, dof, N_dof, N_elements, n_ip, N_nodes
    return None   
#/
def Q8SFMatrix(xi_eta, w2):
    
    #shape function matrix
    N5 = (1/2)*(1 - xi_eta[:, 1])*(1 - xi_eta[:, 0]**2)
    N6 = (1/2)*(1 + xi_eta[:, 0])*(1 - xi_eta[:, 1]**2)
    N7 = (1/2)*(1 + xi_eta[:, 1])*(1 - xi_eta[:, 0]**2)
    N8 = (1/2)*(1 - xi_eta[:, 0])*(1 - xi_eta[:, 1]**2)
    #
    N1 = (1/4)*(1 - xi_eta[:, 0])*(1 - xi_eta[:, 1]) - (1/2)*(N8 + N5)
    N2 = (1/4)*(1 + xi_eta[:, 0])*(1 - xi_eta[:, 1]) - (1/2)*(N5 + N6)
    N3 = (1/4)*(1 + xi_eta[:, 0])*(1 + xi_eta[:, 1]) - (1/2)*(N6 + N7)
    N4 = (1/4)*(1 - xi_eta[:, 0])*(1 + xi_eta[:, 1]) - (1/2)*(N7 + N8)
    #
    N = np.zeros([xi_eta.shape[0],8])
    N[:, 0] = N1
    N[:, 1] = N2
    N[:, 2] = N3
    N[:, 3] = N4
    N[:, 4] = N5
    N[:, 5] = N6
    N[:, 6] = N7
    N[:, 7] = N8
    
    #shape function matrix gradient w.r.t xi_eta[:, 0]
    N5xi = (1/2)*(-2*xi_eta[:, 0] + 2*xi_eta[:, 0]*xi_eta[:, 1])
    N6xi = (1/2)*(1 - xi_eta[:, 1]**2)
    N7xi = (1/2)*(-2*xi_eta[:, 0] - 2*xi_eta[:, 0]*xi_eta[:, 1])
    N8xi = (1/2)*(-1 + xi_eta[:, 1]**2)
    #
    N1xi = (1/4)*(-1 + xi_eta[:, 1]) - (1/2)*(N8xi + N5xi) 
    N2xi = (1/4)*(1 - xi_eta[:, 1]) - (1/2)*(N5xi + N6xi)
    N3xi = (1/4)*(1 + xi_eta[:, 1]) - (1/2)*(N6xi + N7xi)
    N4xi = (1/4)*(-1 - xi_eta[:, 1]) - (1/2)*(N7xi + N8xi)
    
    Nxi = np.zeros([xi_eta.shape[0],8])
    Nxi[:, 0] = N1xi
    Nxi[:, 1] = N2xi
    Nxi[:, 2] = N3xi
    Nxi[:, 3] = N4xi
    Nxi[:, 4] = N5xi
    Nxi[:, 5] = N6xi
    Nxi[:, 6] = N7xi
    Nxi[:, 7] = N8xi

    #shape function matrix gradient w.r.t xi_eta[:, 1]
    N5eta = (1/2)*(-1 + xi_eta[:, 0]**2)
    N6eta = (1/2)*(-2*xi_eta[:, 1] - 2*xi_eta[:, 0]*xi_eta[:, 1])
    N7eta = (1/2)*(1 - xi_eta[:, 0]**2)
    N8eta = (1/2)*(-2*xi_eta[:, 1] + 2*xi_eta[:, 0]*xi_eta[:, 1])
    #
    N1eta = (1/4)*(-1 + xi_eta[:, 0]) - (1/2)*(N8eta + N5eta) 
    N2eta = (1/4)*(-1 - xi_eta[:, 0]) - (1/2)*(N5eta + N6eta)
    N3eta = (1/4)*(1 + xi_eta[:, 0]) - (1/2)*(N6eta + N7eta)
    N4eta = (1/4)*(1 - xi_eta[:, 0]) - (1/2)*(N7eta + N8eta)
    #
    Neta = np.zeros([xi_eta.shape[0],8])
    Neta[:, 0] = N1eta
    Neta[:, 1] = N2eta
    Neta[:, 2] = N3eta
    Neta[:, 3] = N4eta
    Neta[:, 4] = N5eta
    Neta[:, 5] = N6eta
    Neta[:, 6] = N7eta
    Neta[:, 7] = N8eta

    return [N, Nxi, Neta, w2]
#/
def Q4SFMatrix(xi_eta, W2):

    #element shape function matrix
    N = np.zeros([xi_eta.shape[0], 4])
    N1 = (1/4)*(1-xi_eta[:, 0])*(1-xi_eta[:,1])
    N2 = (1/4)*(1+xi_eta[:, 0])*(1-xi_eta[:,1])
    N3 = (1/4)*(1+xi_eta[:, 0])*(1+xi_eta[:,1])
    N4 = (1/4)*(1-xi_eta[:, 0])*(1+xi_eta[:,1])
    N[:, 0] = N1
    N[:, 1] = N2
    N[:, 2] = N3
    N[:, 3] = N4

    #shape function matrix wrt xi
    dNdxsi = np.zeros([xi_eta.shape[0], 4])
    dN1dxsi = (1/4)*(xi_eta[:, 1] - 1)
    dN2dxsi = (1/4)*(-xi_eta[:, 1] + 1)
    dN3dxsi = (1/4)*(xi_eta[:, 1] + 1)
    dN4dxsi = (1/4)*(-xi_eta[:, 1] - 1)
    dNdxsi[:, 0] = dN1dxsi
    dNdxsi[:, 1] = dN2dxsi
    dNdxsi[:, 2] = dN3dxsi
    dNdxsi[:, 3] = dN4dxsi

    #shape function matrix gradient wrt eta
    dNdeta = np.zeros([xi_eta.shape[0], 4])
    dN1deta = (1/4)*(xi_eta[:, 0] - 1)
    dN2deta  = (1/4)*(-xi_eta[:, 0] - 1)
    dN3deta  = (1/4)*(xi_eta[:, 0] + 1)
    dN4deta  = (1/4)*(-xi_eta[:, 0] + 1)
    dNdeta[:, 0] = dN1deta
    dNdeta[:, 1] = dN2deta
    dNdeta[:, 2] = dN3deta
    dNdeta[:, 3] = dN4deta

    return [N, dNdxsi, dNdeta, W2]
#/
def SumSquaredDisplacementGradients(dudx, dudy, dvdx, dvdy, s):

    #area scaling factor stored in diagonal vector
    s = sp.sparse.diags(s)

    Ti = (dudx.T@s@dudx +
        dvdy.T@s@dvdy +
        dudy.T@s@dudy +
        dvdx.T@s@dvdx)

    return Ti
#/
def Hessian(x_vector, y_vector, F_interp, N_global_x, N_global_y, wdetj):

    Xs = x_vector
    Ys = y_vector

    f = np.array([F_interp(Ys, Xs)]).T
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    
    dfdx = np.array([dFdX_interp(Ys, Xs)]).T
    dfdy = np.array([dFdY_interp(Ys, Xs)]).T

    J = (
            sp.sparse.diags(np.squeeze(dfdx))@N_global_x
            + sp.sparse.diags(np.squeeze(dfdy))@N_global_y )

    w_J = sp.sparse.diags(wdetj)@J
    
    H = np.dot(J.T, w_J)

    return [H, f, f_mean, f_tilde, J]
#/
def Residual(F_data, G_interp, x_vector, N_global_x, y_vector, N_global_y, d):
    
    #global image coordinates for the deformed image sampling points
    x, y = x_vector + N_global_x@d, y_vector + N_global_y@d
    
    #reference image data
    f = F_data[1]
    f_mean = F_data[2]
    f_tilde = F_data[3]
    J = F_data[4]

    #sample intensities from deformed image data
    g = np.array([G_interp(y, x)]).T
    g_mean = g.mean()
    g_tilde = np.sqrt(np.sum((g[:]-g_mean)**2))
   
    #compute the zero normalised residual for the current iteration of the node displacments
    res = (f[:]-f_mean-(f_tilde/g_tilde)*(g[:]-g_mean))
    B = J.T@res
    b = B[:, 0]

    return b, res
#/
def InitMeshEntries(mesh, n_ip, n_sf):

    mesh['wdetJj'] = np.zeros(n_ip)
    mesh['rowj'] = np.zeros(n_ip*n_sf, dtype=int)
    mesh['colj'] = np.zeros(n_ip*n_sf, dtype=int)
    mesh['Nj'] = np.zeros(n_ip*n_sf)
    mesh['d_d_dx_j'] = np.zeros(n_ip*n_sf)
    mesh['d_d_dy_j'] = np.zeros(n_ip*n_sf)

    return None
#/
def FieldGradients(N, xn, yn, el_i):
    
    N_ = N[0]
    Nxi = N[1]
    Neta = N[2]
    #Jacobian matrix of the finite element formulation
    #not to be confused with the jacobian matrix of the Gauss-Newton algorithm
    j11 = Nxi@xn[el_i, :]
    j12 = Nxi@yn[el_i, :]
    j21 = Neta@xn[el_i, :]
    j22 = Neta@yn[el_i, :]
    detj = j11*j22 - j12*j21

    #Inverse of Jacobian matrix entries
    #Jacobian matrix inverse
    Ga11 = (j22/detj)[:, np.newaxis]
    Ga12 = (-j12/detj)[:, np.newaxis]
    Ga21 = (-j21/detj)[:, np.newaxis]
    Ga22 = (j11/detj)[:, np.newaxis]
    
    #field derivatives w.r.t x,y, where the field here is phi
    #(see Cook et al eq 6.2.10, the {d} vector is absent here)
    dphidx = Ga11*Nxi + Ga12*Neta
    dphidy = Ga21*Nxi + Ga22*Neta

    return dphidx, dphidy, detj
#/
