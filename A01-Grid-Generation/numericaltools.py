""" 
Generate matrix to perform a spatial derivative on a 
a non uniform grid
"""
from scipy.sparse import csr_matrix
def nufd(x):
    n = len(x)
    h = x[1:]-x[:n-1]
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))
    b0 = (h[0]+h[1])/(h[0]*h[1]) 
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = csr_matrix((val,(row,col)),shape=(n,n))
    return D

import numpy as np
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    '''
    nf = len(a)     # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]

    xc = ac
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    del bc, cc, dc  # delete variables from memory

    return xc

"""
Tools to generate a stretched grid with refinement at both ends
controlled by a tanh function. The grid is symmetrical.
find_gamma determines the stretching of the grid to achieve
a prescribed grid step at either boundary.
"""
from scipy.optimize import fsolve
def find_gamma(ly,y_uni,dy_min,g_ini):
    def delta_tanh(g):
        return ly/2.0*np.tanh(g*(y_uni[1]))/np.tanh(g*ly/2.0)-(-ly/2+dy_min)
    g_dy_min = fsolve(delta_tanh,g_ini)
    return g_dy_min
def stretched_mesh(ly,ny,dy_min,g_ini):
    y_uni = np.linspace(-ly/2, ly/2, ny)
    gamma_y = find_gamma(ly,y_uni,dy_min,g_ini)
    y_s = ly/2.0*np.tanh(gamma_y*(y_uni))/np.tanh(gamma_y*ly/2.0)
    return y_s,gamma_y


def diffusion_matrix_coefficients(alpha,dt_2,a_metrics,c_metrics):
    """ arguments must be from bottom wall to top wall of dimensions N+2
        returns a,b,c of dimensions N (from first to last points off the walls)"""
    n = len(alpha)
    a = np.zeros(n-2)
    b = np.zeros(n-2)
    c = np.zeros(n-2)
    a[:] = (alpha[0:-2] + alpha[1:-1])*a_metrics[:]
    c[:] = (alpha[2:] + alpha[1:-1])*c_metrics[:]
    b = -(a+c)
    a *= dt_2
    b *= dt_2
    c *= dt_2
    return a,b,c
def rhs_T(a_rhs,b_rhs,c_rhs,a_lhs,c_lhs,T_old_all,T_all,dt_2,dt):
    d = a_rhs*T_old_all[:-2] + (b_rhs + 1)*T_old_all[1:-1] + c_rhs*T_old_all[2:]
    d[0] += a_lhs[0]*T_all[0]
    d[-1] += c_lhs[-1]*T_all[-1]
    return d
