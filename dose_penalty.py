# -*- coding: utf-8 -*-
"""
This module solves the parabolic dose penalized problem
    min_u 1/2 |u|_L2^2 + beta_1 |(Cy-U)^-|_L1(omT) + beta_2|(Cy-L)^+|_L1(omR)
    s.t.  y_t - c\Delta y = u,  y(0) = 0 with hom. Dirichlet b.c.
using a semismooth Newton method as described in the paper
    'L1 penalization of volumetric dose objectives in optimal control of PDEs'
by Richard C. Barnard and Christian Clason, http://arxiv.org/abs/1607.01655
"""

__author__ = "Richard C. Barnard <barnardrc@ornl.gov>", \
             "Christian Clason <christian.clason@uni-due.de>"
__date__ = "July 6, 2016"

import numpy as np
from numpy import matlib
import scipy.sparse as sp
from scipy.sparse import linalg as la
import matplotlib.pyplot as plt

# problem parameters
beta1 = 1e3/0.5   # dose penalty on tumor (per unit volume)
beta2 = 1e4/0.7   # dose penalty on oar (per unit volume)
U     = 5.e-1     # threshold level: tumor
L     = 2.e-1     # threshold level: OAR
maxit = 100       # max number of SSN iterations

# pde parameters
T = 1.            # Final time for PDE solve
nt = 256          # number of time steps
nx = 256          # number nodes each direction
c = 0.01          # diffusion coefficient 

# setup grid
x = np.linspace(-1,1,num = nx)
t = np.linspace(0,T,num =nt)
xx,tt = np.meshgrid(x,t)
h = x[1]-x[0]     # spatial mesh size
tau = t[1]-t[0]   # temporal mesh size

# construct differential operators
ex = np.ones((nx))
et = np.ones((nt))
D2 = sp.diags([(-1./h)*ex[0:nx-1],(2./h)*ex,(-1./h)*ex[0:nx-1]],[-1,0,1])
Mx = sp.diags([(h/6.)*ex[0:nx-1],(2.*h/3.)*ex,(h/6.)*ex[0:nx-1]],[-1,0,1])
Dt = sp.diags([(-1./tau)*et[0:nt-1],(1./tau)*et],[-1,0])
It = sp.eye(nt)
# parabolic diff.op. and mass matrix (space-time form)
A = sp.csc_matrix(sp.kron(Dt,Mx) + c*sp.kron(It,D2))
M = sp.csc_matrix(sp.kron(It,Mx))
# dose operator
C = sp.csr_matrix(matlib.repmat(tau*np.eye(nx),1,nt))

# indicator functions of control, tumor, risk domains 
om_C = np.reshape((xx<-.75).astype('float')+(xx>.75).astype('float'),nx*nt)
om_C = sp.diags(om_C,0)    # control domain
om_T = ((x > -0.45).astype('float')*(x < 0.45).astype('float'))
om_T -= (np.abs(x)<.2).astype('float')
om_T = sp.diags(om_T,0)    # tumor region
om_R = ((x < -0.55).astype('float')*(x > -0.7).astype('float'))
om_R += ((x > 0.55).astype('float')*(x < 0.7).astype('float'))
om_R += (np.abs(x)<.2).astype('float')
om_R = sp.diags(om_R,0)    # risk region

# closure of forward operators, adjoints
solve = la.factorized(A)
solve_adjoint = la.factorized(sp.csc_matrix(A.transpose()))
def S(u):
    return np.reshape(solve(M*u),u.shape)
def St(r):
    return np.reshape(M*solve_adjoint(r),r.shape)
def doseT(y):
    return om_T*(C*y)
def doseTt(y):
    return C.T*(om_T.T*y)
def doseR(y):
    return om_R*(C*y)
def doseRt(y):
    return C.T*(om_R.T*y)

# closure of regularized subdifferentials, Newton derivatives
def H_plus(v,gamma):
    return 1./gamma*(v-L)*((v>=L)&(v<=L+gamma)) + 1.0*(v>L+gamma)
def H_minus(v,gamma):
    return 1./gamma*(v-U)*((v<=U)&(v>=U-gamma)) - 1.0*(v<U-gamma)
def DH_plus(v,gamma):
    return 1./gamma*((v>=L)&(v<=L+gamma))
def DH_minus(v,gamma):
    return 1./gamma*((v<=U)&(v>=U-gamma))

def F_uk(u):
    """Evaluate gradient (w/o projection)"""
    y = S(u)
    prox =  beta1*doseTt(Mx*H_minus(doseT(y),gamma))
    prox += beta2*doseRt(Mx*H_plus(doseR(y),gamma))
    return St(prox),y

def DNF_uk(du,uk,yk,gamma):
    """Apply Newton derivative (w/o projection)"""
    Sdu = S(du)
    prox =  beta1*doseTt(DH_minus(doseT(yk),gamma)*Mx*doseT(Sdu))
    prox += beta2*doseRt(DH_plus(doseR(yk),gamma)*Mx*doseR(Sdu))
    return St(prox)

def comp_DVH(dose):
    """Compute dose volumetric histogram"""
    DVH = np.zeros((2,200))
    doses = np.linspace(0.,1.2*np.maximum(U,L),num = 200)
    for i in range(200):
        voxels = np.where(om_R*dose>doses[i])
        DVH[0,i] = 1.*voxels[0].size/np.sum(om_R.toarray())
        voxels = np.where(om_T*dose>doses[i])
        DVH[1,i] = 1.*voxels[0].size/np.sum(om_T.toarray())        
    return DVH,doses

def SSN_loop(u,gamma):
    """semismooth Newton method for fixed gamma and starting point u"""
    # compute gradient
    Fuk,y  = F_uk(u)
    grad = M*u-np.maximum(-1.*Fuk,0)
    gradNorm = np.dot(grad.T,M*grad)
    firstNorm = np.copy(gradNorm)
    k = 0
    consec = 0
    while (k<maxit):
        print 'It# %d: residual = %1.3e' % (k,gradNorm)
        k += 1
        # application of Newton derivative, solve for Newton step
        Hdu = lambda du: M*du + (-Fuk>=0)*DNF_uk(du,u,y,gamma)
        H = la.LinearOperator((nx*nt,nx*nt), matvec = Hdu, dtype = 'float')
        du,flag = la.gmres(H,-grad,x0=None,restart=3000,maxiter=3000,tol=1e-9)
        if (flag):
            print "Warning, GMRES did not fully converge"
        # perform linesearch
        delta = 1.
        while(delta>=1.e-6):
            tmpu = u + delta*du
            Fuk,tmpy  = F_uk(tmpu)
            grad = M*tmpu-np.maximum(-1.*Fuk,0)
            tmpNorm = np.dot(grad.T,M*grad)
            if (tmpNorm<gradNorm):
                u = np.copy(tmpu)
                y = np.copy(tmpy)
                gradNorm = np.copy(tmpNorm)
                if(consec>0):
                    if (np.dot(delta*du,M*(delta*du))>1.e-6):
                        consec += 1
                    else:
                        consec -= 1
                break
            else:
                delta *= .5
        if (delta<1.e-6):
            print "Step size too small, accepting ascent step"
            u = np.copy(tmpu)
            gradNorm = np.copy(tmpNorm)
            consec += 1
        if (gradNorm<1.e-6):
            print "Sufficient decrease in gradient norm, terminating"
            break
        if (consec>4):
            print "Terminating due to too many bad consecutive steps"
            break

    # compute statistics, output
    DVH,levels = comp_DVH(C*y)
    lowlevel = np.where(levels-L>0)[0][0]
    uplevel = np.where(levels-U>0)[0][0]    
    print ('It# %d: above L=%1.3e, below U=%1.3e, residual = %1.3e'
                % (k,DVH[0,lowlevel],1-DVH[1,uplevel],gradNorm))
    return u,y,gradNorm


# initialize plots
plt.ion()
fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()  

# homotopy loop
u = np.zeros(nx*nt)
gamma = 1.e+2
while (gamma>1e-7):
    print "\nGamma = %1.2e" % gamma
    u,y,residual = SSN_loop(u,gamma)
    if residual > 1e-6:
        break
    else:
        # plot results
        DVH,levels = comp_DVH(C*y)
        ax1.cla()
        ax1.plot(x,C*y, label = 'Cy')
        ax1.plot(x,np.diagonal(om_T.toarray())*U,label='U \chi_{\omega_T}')
        ax1.plot(x,np.diagonal(om_R.toarray())*L,label='L \chi_{\omega_R}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Dose')
        ax1.legend()
        plt.pause(0.0001)
        ax2.cla()
        ax2.plot(levels,DVH[0,:],color='r',label='Risk')
        ax2.plot(levels,DVH[1,:],color='g',label='Target')
        ax2.axvline(x=L,color='r',linestyle='dashed')
        ax2.axvline(x=U,color='g',linestyle='dashed')
        ax2.set_xlabel('Dose Level')
        ax2.set_ylabel('Volume Fraction')
        ax2.legend()
        plt.pause(0.0001)
        # update gamma
        gamma = gamma/2.
