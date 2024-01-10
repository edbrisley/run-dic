import numpy as np
from fast_interp import interp2d
import cv2 as cv
import scipy as sp
from numpy import copy
import scipy.sparse.linalg as splalg
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from operator import itemgetter
import GQ_rosettacode_Python3 as gq
#/
def FastInterpolation(image):

    #image coordinates
    ny = image.shape[0]
    nx = image.shape[1]

        #interpolation
    image_interpolated = interp2d([0,0], [ny-1,nx-1], [1,1], image, k=3, p=[False,False], e=[1,0])
    return image_interpolated
#/
def ProcessImageData(F, G):

    data = dict()

    #reference image data, processing, interpolation
    F_interp = FastInterpolation(F)
    dFdY, dFdX = np.array(np.gradient(F), dtype=float)
    dFdX_interp = FastInterpolation(dFdX)
    dFdY_interp = FastInterpolation(dFdY)

    #deformed image interpolation
    G_interp = FastInterpolation(G)

    data = dict()
    data['F_interp'] = F_interp
    data['dFdY'] = dFdY
    data['dFdX'] = dFdX
    data['dFdX_interp'] = dFdX_interp
    data['dFdY_interp'] = dFdY_interp
    data['G_interp'] = G_interp

    return data
#/
def FEMesh(q_rule, el_type, el_size, alpha2, ROIcoords):

    #create the mesh dictionary and the associated shape function data
    mesh = QuadrilateralSF(q_rule, el_type, el_size, alpha2, ROIcoords)
    #mesh shape functions together and prepare the data for global assembly
    QuadrilateralMesh(mesh)
    #assemble the finite elments and associated data to the global mesh
    GlobalAssembly(mesh)

    return mesh
#/
def QuadrilateralSF(q_rule, el_type, el_size, alpha2, ROIcoords):

    mesh = dict()
    #user-defined settings
    mesh['el_size'] = el_size
    mesh['alpha2'] = alpha2
    mesh['ROIcoords'] = ROIcoords
    mesh['el_type'] = el_type

    #external call to RosettaCode library to return the
    #interpolation point coordinates and the quadrature weights
    xsi_eta, w2 = GaussQuadrature2D(q_rule)

    #shape function matrix for a single element
    if el_type == 'Q8':
        N  = Q8SFMatrix(xsi_eta, w2)
    elif el_type == 'Q4':
        N  = Q4SFMatrix(xsi_eta, w2)
    else:
        print('element type not supported')
        pass

    n_sf = N[0].shape[1]
    n_q = N[0].shape[0]
    wg = N[3]

    mesh['N'] = N
    mesh['wg'] = wg
    mesh['n_q'] = n_q
    mesh['n_sf'] = n_sf

    return mesh
#/
def QuadrilateralMesh(mesh):

    if mesh['el_type'] == 'Q8':
        Q8RectangularMesh(mesh)
    elif mesh['el_type'] == 'Q4':
        Q4RectangularMesh(mesh)
    
    return None
#/
def Q8RectangularMesh(m):

    """Q8 mesh for rectangular region-of-interest"""
    #-------------------------------------------------------------------------------------
    #ROI boundary coordinates
    x_start, x_end, y_start, y_end = m['ROIcoords'][0], m['ROIcoords'][1], m['ROIcoords'][2], m['ROIcoords'][3]
    #dummy grid to count number of rows and columns in mesh
    dummy_node_grid = np.meshgrid(np.arange(y_start,
                                            y_end,
                                            (m['el_size']-1)/2),
                                    np.arange(x_start,
                                            x_end,
                                            (m['el_size']-1)/2),
                                            indexing = 'ij')
    #count number of rows and columns in node grid
    N_node_rows = dummy_node_grid[0].shape[0]
    N_node_cols = dummy_node_grid[0].shape[1]
    
    #mesh 
    n_elements = int(((N_node_rows -1)/2)*((N_node_cols-1)/2))
    n_nodes = (N_node_cols*N_node_rows - n_elements)
    n_ip = m['n_q']*n_elements 
    n_dof = int(2*n_nodes)
    
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
    element_conn = np.zeros([n_elements, 8]).astype(int)
    start = 0
    for col in range(0, int((N_node_cols-1)/2)):
        for row in range(0, int((N_node_rows-1)/2)):
            element_conn[int(start+row), :] = el1 + row*np.array([2, 2, 2, 2, 2, 1, 2, 1])
        #increment element info
        start = start + (N_node_rows-1)/2
        el1 = el1 + (1+(3*N_node_rows-1)/2)
    
    #node coordinates sub-function
    node_coords = np.zeros([n_nodes, 2])
    y_odd = np.arange(y_start, y_end, (m['el_size']-1))
    row_start = 0
    col_start = 0
    for x in range(0, N_node_cols):
        if x%2 == 0:
            node_coords[row_start:row_start + N_node_rows, :] = np.vstack(( np.repeat(x_start + x*(m['el_size']-1)/2, N_node_rows),
                                                                            np.arange(y_start, y_end, (m['el_size']-1)/2))).T
            row_start = row_start + N_node_rows
        else:
            node_coords[row_start:row_start + int((N_node_rows-1)/2 + 1), :] = np.vstack((  np.repeat(x_start + x*(m['el_size']-1)/2,
                                                                                                    (N_node_rows-1)/2 + 1),

                                                                                            y_odd)).T
            row_start = int(row_start + (N_node_rows-1)/2 + 1)
    
    #concatenate x-dofs and y-dofs
    x_dofs = np.array(np.arange(n_nodes))
    y_dofs = np.array(n_nodes + np.arange(n_nodes))
    dof = np.vstack(( x_dofs,
                      y_dofs )).T

                      #node x and y coordinates and x-dofs in mesh
    xn = node_coords[element_conn, 0]
    yn = node_coords[element_conn, 1]
    x_dofs_mesh = dof[element_conn, 0]
    
    m['node_coords'] = node_coords
    m['element_conn'] = element_conn
    m['dof'] = dof
    m['n_nodes'] = n_nodes
    m['n_elements'] = n_elements
    m['n_ip'] = n_ip
    m['n_dof'] = n_dof

    m['xn'] = xn
    m['yn'] = yn
    m['x_dofs_mesh'] = x_dofs_mesh

    #return node_coords, element_conn, dof, N_dof, n_elements, n_ip, n_nodes
    return None
#/
def Q4RectangularMesh(m):
    "FE mesh for Q4 element type"

    x_start, x_end, y_start, y_end = m['ROIcoords'][0], m['ROIcoords'][1], m['ROIcoords'][2], m['ROIcoords'][3]
    #dummy grid to count number of rows and columns in mesh
    nodes_y, nodes_x = np.meshgrid(np.arange(y_start,
                                            y_end,
                                            (m['el_size']-1)/2),
                                    np.arange(x_start,
                                            x_end,
                                            (m['el_size']-1)/2),
                                            indexing = 'ij')

    #count the number of rows and columns in the mesh
    N_node_rows = nodes_y.shape[0]
    N_node_cols = nodes_y.shape[1]
    n_elements = (N_node_cols-1)*(N_node_rows-1)

    #vectors
    node_coords = np.vstack((np.array([nodes_x.flatten(order = 'F')]), np.array([nodes_y.flatten(order = 'F')])))
    n_nodes = node_coords.shape[1]
    #nodes xy coordinates
    node_coords= node_coords.T

    #define the node numbers as they appear in the mesh
    mesh_node_no = np.arange(0, n_nodes).reshape(N_node_rows,
                                                N_node_cols,
                                                order = 'F')
    
    #element connectivity table
    element_conn = np.zeros([n_elements, 4]).astype(int)
    l = 0
    for j in range(0, N_node_cols - 1):
        #rows
        for i in range(0, N_node_rows - 1):
            element_conn[l, :] = np.array([mesh_node_no[i,j], mesh_node_no[i+1, j], mesh_node_no[i+1, j+1], mesh_node_no[i, j+1]])
            l = l + 1
    
    #connectivity
    dof = np.arange(n_nodes)
    dof = np.c_[dof, dof + n_nodes*(dof>=0)]
    n_dof = 2*n_nodes
    n_ip = m['n_q']*n_elements

    #node x and y coordinates and x-dofs in mesh
    xn = node_coords[element_conn, 0]
    yn = node_coords[element_conn, 1]
    x_dofs_mesh = dof[element_conn, 0]

    m['node_coords'] = node_coords
    m['element_conn'] = element_conn
    m['dof'] = dof
    m['n_nodes'] = n_nodes
    m['n_elements'] = n_elements
    m['n_ip'] = n_ip
    m['n_dof'] = n_dof

    m['xn'] = xn
    m['yn'] = yn
    m['x_dofs_mesh'] = x_dofs_mesh

    #return node_coords, element_conn, dof, N_dof, n_elements, n_ip, n_nodes
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
def SumSquaredDisplacementGradients(m):

    #area scaling factor stored in diagonal vector
    #moving from original to natural coordinate systems
    s = sp.sparse.diags(m['wdetj'])

    Ti = (m['dudx_'].T@s@m['dudx_'] +
        m['dvdy_'].T@s@m['dvdy_'] +
        m['dudy_'].T@s@m['dudy_'] +
        m['dvdx_'].T@s@m['dvdx_'])
    
    m['Ti'] = Ti

    return None
#/
def Hessian(m, data):

    #sampling point x and y coordinates
    Xs, Ys = m['x_vector'], m['y_vector']

    F_interp, dFdY_interp, dFdX_interp = data['F_interp'], data['dFdY_interp'], data['dFdX_interp']

    f = np.array([F_interp(Ys, Xs)]).T
    f_mean = f.mean()
    f_tilde = np.sqrt(np.sum((f[:]-f_mean)**2))
    
    dfdy = np.array([dFdY_interp(Ys, Xs)]).T
    dfdx = np.array([dFdX_interp(Ys, Xs)]).T
    
    J = (
            sp.sparse.diags(np.squeeze(dfdx))@m['N_global_x']
            + sp.sparse.diags(np.squeeze(dfdy))@m['N_global_y'] )

    w_J = sp.sparse.diags(m['wdetj'])@J
    
    H = np.dot(J.T, w_J)

    data['H'], data['J'], data['f'], data['f_mean'], data['f_tilde'] = H, J, f, f_mean, f_tilde

    return None
#/
def Residual(m, data, d):
    
    #global image coordinates for the deformed image sampling points
    x, y = m['x_vector'] + m['N_global_x']@d, m['y_vector'] + m['N_global_y']@d
    
    #reference image data
    f = data['f']
    f_mean = data['f_mean']
    f_tilde = data['f_tilde']
    J = data['J']

    #sample intensities from deformed image data
    g = np.array([data['G_interp'](y, x)]).T
    g_mean = g.mean()
    g_tilde = np.sqrt(np.sum((g[:]-g_mean)**2))
   
    #compute the zero normalised residual for the current iteration of the node displacments
    res = (f[:]-f_mean-(f_tilde/g_tilde)*(g[:]-g_mean))
    B = J.T@res
    b = B[:, 0]

    return b, res
#/
def InitMeshEntries(mesh):

    #initialize global arrays
    n_ip, n_sf = mesh['n_ip'], mesh['n_sf']

    mesh['wdetj'] = np.zeros(n_ip)
    mesh['rowj'] = np.zeros(n_ip*n_sf, dtype=int)
    mesh['colj'] = np.zeros(n_ip*n_sf, dtype=int)
    mesh['Nj'] = np.zeros(n_ip*n_sf)
    #the 2D-displacment field is referred to here as phi.. phi = [u, v]
    mesh['dphidx_j'] = np.zeros(n_ip*n_sf)
    mesh['dphidy_j'] = np.zeros(n_ip*n_sf)

    #mesh dictionary is passed by reference
    return None
#/
def FieldGradients(m, el_i):
    
    #extract shape function matrix data
    N_ = m['N'][0]
    Nxi = m['N'][1]
    Neta = m['N'][2]

    #Jacobian matrix of the finite element formulation
    #not to be confused with the jacobian matrix of the Gauss-Newton algorithm
    j11 = Nxi@m['xn'][el_i, :]
    j12 = Nxi@m['yn'][el_i, :]
    j21 = Neta@m['xn'][el_i, :]
    j22 = Neta@m['yn'][el_i, :]
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

    # m['dphidx_j'] = dphidx 
    # m['dphidy_j'] = dphidy
    # m['detj'] = detj

    return dphidx, dphidy, detj
#/
def GaussQuadrature2D(order):

    xsi_eta, w2 = gq.ExportGQrundic(order)

    return xsi_eta, w2
#/
def GlobalAssembly(m):

    #initialize global arrays, initialize gloabal array for single element type (SubMesh)
    InitMeshEntries(m)
    #assemble the indidivudal elements to the global mesh arrays
    AssembleIndividualElements(m)
    #Reshape/store the individual element entries to scipy sparse matrices
    #this saves
    ReshapetoSparse(m)

    return None
#/
def AssembleIndividualElements(m):

    index0 = 0
    for el_i in range(m['n_elements']):
        
        #current element indices for assembly to global matrix
        indices_el = index0 + np.arange(m['n_q'])
        ind = m['n_sf']*index0 + np.arange(m['n_q']*8)
        [el_cols, el_rows] = np.meshgrid(m['x_dofs_mesh'][el_i, :], indices_el)
        m['rowj'][ind] = el_rows.ravel()
        m['colj'][ind] = el_cols.ravel()
        m['Nj'][ind] = m['N'][0].ravel()

        #displacement gradients current element
        dphidx, dphidy, detJ = FieldGradients(m, el_i)
        
        #assemble derivatives to global matrix
        m['wdetj'][indices_el] = m['wg']*abs(detJ)
        m['dphidx_j'][ind] = dphidx.ravel()
        m['dphidy_j'][ind] = dphidy.ravel()

        index0 += m['n_q']

    return None

def ReshapetoSparse(m):

    N_global_x = sp.sparse.csc_matrix(
                                        (m['Nj'], (m['rowj'], m['colj'])),
                                        shape=(m['n_ip'], m['n_dof']))
    N_global_y = sp.sparse.csc_matrix(
                                        (m['Nj'], (m['rowj'], m['colj'] + m['n_dof']//2)),
                                        shape=(m['n_ip'], m['n_dof']))
    dudx_ = sp.sparse.csc_matrix(
                                    (m['dphidx_j'], (m['rowj'], m['colj'])),
                                    shape=(m['n_ip'], m['n_dof']))
    dudy_ = sp.sparse.csc_matrix(
                                    (m['dphidy_j'], (m['rowj'], m['colj'])),
                                    shape=(m['n_ip'], m['n_dof']))
    dvdx_ = sp.sparse.csc_matrix(
                                    (m['dphidx_j'],(m['rowj'], m['colj'] + m['n_dof']//2)),
                                    shape=(m['n_ip'], m['n_dof']))
    dvdy_ = sp.sparse.csc_matrix(
                                    (m['dphidy_j'], (m['rowj'], m['colj'] + m['n_dof']//2)),
                                    shape=(m['n_ip'], m['n_dof']))    
    
    #store to mesh dictionary
    m['xcoords'] = m['node_coords'][:, 0][m['dof'][:, 0]]
    m['ycoords'] = m['node_coords'][:, 1][m['dof'][:, 0]]
    m['xcoords_append_ycoords'] = np.hstack((m['xcoords'], m['ycoords']))

    x_vector = N_global_x.dot(m['xcoords_append_ycoords'])
    y_vector = N_global_y.dot(m['xcoords_append_ycoords'])

    m['N_global_x'] = N_global_x
    m['N_global_y'] = N_global_y

    m['dudx_'], m['dudy_'], m['dvdx_'], m['dvdy_'] = dudx_, dudy_, dvdx_, dvdy_
    m['x_vector'], m['y_vector'], m['x_vector'], m['y_vector'] = x_vector, y_vector, x_vector, y_vector

    return None

def RegularizedModifiedGN(m, data, d0):

    #initial estimate of nodal displacments
    d = copy(d0)
    #assemble the Hessian, Jacobian matrices from the reference image data
    Hessian(m, data)
    #
    SumSquaredDisplacementGradients(m)
    H = data['H']
    alpha2, Ti = m['alpha2'], m['Ti']
    #regularize the Hessian matrix
    H_reg = H.T@H + alpha2*Ti

    #perform LU factorisation of the regularized Hessian matrix
    #note that after regularization, much of the sparseness in the
    #Hessian is lost, making LU a viable factorization method here
    #it is also currenlty the only supported factorization method
    #in the scipy sparse library (would like to try Cholesky when they add it)
    H_LU = splalg.splu(H_reg)

    #perform the NL least-squares minimization
    #i.e solve the regularized, modified GN linear system
    #of equations at each iteration
    for k in range(0,100):

        #compute zero-normalised residual and the right-hand-side operand
        b, res = Residual(m, data, d)
        
        #compute the incremental update to the nodal
        #displacement vector
        delta_d = H_LU.solve(H.T@b)

        #compute the euclidian difference between nodal displacements
        #between the iterations 
        err = np.linalg.norm((d + delta_d) - d)
        if err<1e-3:
            break
        d += delta_d
        
        print("Iteration: {}, euclidian difference: {}".format(k, err))
        

    return d

