from numpy import *
import numpy as np
 
#This library was copied and pasted from https://rosettacode.org/wiki/Numerical_integration/Gauss-Legendre_Quadrature#Python
#The library was then adapted to use Python 3 instead of Python 2

#################################################################
# Recursive generation of the Legendre polynomial of order n
def Legendre(n,x):
	x=array(x)
	if (n==0):
		return x*0+1.0
	elif (n==1):
		return x
	else:
		return ((2.0*n-1.0)*x*Legendre(n-1,x)-(n-1)*Legendre(n-2,x))/n
 
##################################################################
# Derivative of the Legendre polynomials
def DLegendre(n,x):
	x=array(x)
	if (n==0):
		return x*0
	elif (n==1):
		return x*0+1.0
	else:
		return (n/(x**2-1.0))*(x*Legendre(n,x)-Legendre(n-1,x))
##################################################################
# Roots of the polynomial obtained using Newton-Raphson method
def LegendreRoots(polyorder,tolerance=1e-20):
	if polyorder<2:
		err=1 # bad polyorder no roots can be found
	else:
		roots=[]
		# The polynomials are alternately even and odd functions. So we evaluate only half the number of roots. 
		for i in range(1,int(polyorder/2) +1):
			x=cos(pi*(i-0.25)/(polyorder+0.5))
			error=10*tolerance
			iters=0
			while (error>tolerance) and (iters<1000):
				dx=-Legendre(polyorder,x)/DLegendre(polyorder,x)
				x=x+dx
				iters=iters+1
				error=abs(dx)
			roots.append(x)
		# Use symmetry to get the other roots
		roots=array(roots)
		if polyorder%2==0:
			roots=concatenate( (-1.0*roots, roots[::-1]) )
		else:
			roots=concatenate( (-1.0*roots, [0.0], roots[::-1]) )
		err=0 # successfully determined roots
	return [roots, err]
#################################################################
# Weight coefficients
def GaussLegendreWeights(polyorder):
	W=[]
	[xis,err]=LegendreRoots(polyorder)
	if err==0:
		W=2.0/( (1.0-xis**2)*(DLegendre(polyorder,xis)**2) )
		err=0
	else:
		err=1 # could not determine roots - so no weights
	return [W, xis, err]
##################################################################
# The integral value 
# func 		: the integrand
# a, b 		: lower and upper limits of the integral
# polyorder 	: order of the Legendre polynomial to be used
#
def GaussLegendreQuadrature(func, polyorder, a, b):
	[Ws,xs, err]= GaussLegendreWeights(polyorder)
	if err==0:
		ans=(b-a)*0.5*sum( Ws*func( (b-a)*0.5*xs+ (b+a)*0.5 ) )
	else: 
		# (in case of error)
		err=1
		ans=None
	return [ans,err]
##################################################################


# The integrand - change as required
def func(x):
	return exp(x)
##################################################################

def Q8SFMatrix():
    
    #1D sampling point absolute value
    
    #original quadrature rule
    coords1 = np.array([-0.97390653, -0.86506337, -0.67940957, -0.43339539, -0.14887434 ,
                     0.14887434, 0.43339539, 0.67940957, 0.86506337 , 0.97390653])

    g_y, g_x = np.meshgrid(coords1,
                            coords1,
                            indexing = 'ij')
    N_coords_y = g_y.shape[0]
    N_coords_x = g_y.shape[1]
    #vectors
    el_coords = np.vstack((np.array([g_x.flatten(order = 'F')]), np.array([g_y.flatten(order = 'F')])))
    N_p = el_coords.shape[1]
    #nodes xy coordinates
    xi_eta = el_coords.T

    ###
    weights1 = np.array([0.06667134, 0.14945135, 0.21908636, 0.26926672, 0.29552422, 0.29552422,
                            0.26926672, 0.21908636, 0.14945135, 0.06667134])

    w_y, w_x = np.meshgrid( weights1,
                            weights1,
                            indexing = 'ij' )
    N_w_y = w_y.shape[0]
    N_w_x = w_y.shape[1]
    #vectors
    w_coords = np.vstack((np.array([w_x.flatten(order = 'F')]), np.array([w_y.flatten(order = 'F')])))
    N_p = w_coords.shape[1]
    #nodes xy coordinates
    wg = w_coords.T

    w2 = wg[:, 0]*wg[:, 1]
    
    #shape function matrix
    N1 = -(1/4)*(1 - xi_eta[:, 0])*(1 - xi_eta[:, 1])*(1 + xi_eta[:, 0] + xi_eta[:, 1])
    N2 = -(1/4)*(1 + xi_eta[:, 0])*(1 - xi_eta[:, 1])*(1 - xi_eta[:, 0] + xi_eta[:, 1])
    N3 = -(1/4)*(1 + xi_eta[:, 0])*(1 + xi_eta[:, 1])*(1 - xi_eta[:, 0] - xi_eta[:, 1])
    N4 = -(1/4)*(1 - xi_eta[:, 0])*(1 + xi_eta[:, 1])*(1 + xi_eta[:, 0] - xi_eta[:, 1])
    N5 = (1/2)*(1 - xi_eta[:, 1])*(1 - xi_eta[:, 0]**2)
    N6 = (1/2)*(1 + xi_eta[:, 0])*(1 - xi_eta[:, 1]**2)
    N7 = (1/2)*(1 + xi_eta[:, 1])*(1 - xi_eta[:, 0]**2)
    N8 = (1/2)*(1 - xi_eta[:, 0])*(1 - xi_eta[:, 1]**2)
    N = np.zeros([100,8])
    N[:, 0] = N1
    N[:, 1] = N2
    N[:, 2] = N3
    N[:, 3] = N4
    N[:, 4] = N5
    N[:, 5] = N6
    N[:, 6] = N7
    N[:, 7] = N8
    
    #shape function matrix gradient w.r.t xi_eta[:, 0]
    N1xi = -(1/4)*(-1 + xi_eta[:, 1])*(2 * xi_eta[:, 0] + xi_eta[:, 1])
    N2xi = (1/4)*(-1 + xi_eta[:, 1])*(xi_eta[:, 1] - 2 * xi_eta[:, 0])
    N3xi = (1/4)*(1 + xi_eta[:, 1])*(2 * xi_eta[:, 0] + xi_eta[:, 1])
    N4xi = -(1/4)*(1 + xi_eta[:, 1])*(xi_eta[:, 1] - 2 * xi_eta[:, 0])
    N5xi = -xi_eta[:, 0]*(1 - xi_eta[:, 1])
    N6xi = -(1/2)*(1 + xi_eta[:, 1])*(-1 + xi_eta[:, 1])
    N7xi = -xi_eta[:, 0]*(1 + xi_eta[:, 1])
    N8xi = -(1/2)*(1 + xi_eta[:, 1])*(1 - xi_eta[:, 1])
    Nxi = np.zeros([100,8])
    Nxi[:, 0] = N1xi
    Nxi[:, 1] = N2xi
    Nxi[:, 2] = N3xi
    Nxi[:, 3] = N4xi
    Nxi[:, 4] = N5xi
    Nxi[:, 5] = N6xi
    Nxi[:, 6] = N7xi
    Nxi[:, 7] = N8xi

    #shape function matrix gradient w.r.t xi_eta[:, 1]
    N1eta = -(1/4)*(-1 + xi_eta[:, 0])*(xi_eta[:, 0] + 2*xi_eta[:, 1])
    N2eta = (1/4)*(1 + xi_eta[:, 0])*(2*xi_eta[:, 1] - xi_eta[:, 0])
    N3eta = (1/4)*(1 + xi_eta[:, 0])*(xi_eta[:, 0] + 2*xi_eta[:, 1])
    N4eta = -(1/4)*(-1 + xi_eta[:, 0])*(2*xi_eta[:, 1] - xi_eta[:, 0])
    N5eta = (1/2)*(1 + xi_eta[:, 0])*(-1 + xi_eta[:, 0])
    N6eta = -xi_eta[:, 1]*(1 + xi_eta[:, 0])
    N7eta = -(1/2)*(1 + xi_eta[:, 0])*(-1 + xi_eta[:, 0])
    N8eta = xi_eta[:, 1]*(-1 + xi_eta[:, 0])
    Neta = np.zeros([100,8])
    Neta[:, 0] = N1eta
    Neta[:, 1] = N2eta
    Neta[:, 2] = N3eta
    Neta[:, 3] = N4eta
    Neta[:, 4] = N5eta
    Neta[:, 5] = N6eta
    Neta[:, 6] = N7eta
    Neta[:, 7] = N8eta

    return [N, Nxi, Neta, w2]
#
def ExportGQrundic(order):

	#1D quadrature weights and points
	[w_1D, xsi_1D, err] = GaussLegendreWeights(order)
	w_1D = np.array(w_1D)
	xsi_1D = np.array(xsi_1D)

	#2D quadrature points
	xsi, eta = np.meshgrid( xsi_1D,
                            xsi_1D,
                            indexing = 'ij' )

	xsi_eta = np.vstack(( xsi.ravel(order = 'F'), eta.ravel(order = 'F') )).T

	#2D quadrature weights
	w_x, w_y = np.meshgrid( w_1D,
                            w_1D,
                            indexing = 'ij' )

	wg = np.vstack((w_x.ravel(order = 'F'), w_y.ravel(order = 'F'))).T
	
	#product of weights
	w2 = wg[:, 0]*wg[:, 1]

	return xsi_eta, w2


##################################################################

#Example: Uncomment all code below and run python script to illu-
#strate the integration of e^x over the domain [-3, 3] 



# order=10
# [Ws,xs,err] = GaussLegendreWeights(order)
# if err == 0:
# 	print ("Order    : ", order)
# 	print ("Roots    : ", xs)
# 	print ("Weights  : ", Ws)
# 	print('No. of roots/weights', xs.shape, Ws.shape)
# 	w = np.array(Ws)
# 	print(w)
	
# else:
# 	print ("Roots/Weights evaluation failed")
 
# # Integrating the function
# [ans,err] = GaussLegendreQuadrature(func , order, -3,3)
# if err == 0:
# 	print ("Integral : ", ans)
# else:
# 	print ("Integral evaluation failed")

