# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:55:44 2025

@author: Harry MullineauxSanders
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import itertools as itr
from scipy.optimize import fsolve
from random import uniform
from pfapack import pfaffian as pf


plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


#Hamiltonians
def QWZ_model(kx,ky,mu,t,a):
    H=np.array(([mu-t*(np.cos(kx)+np.cos(ky)),a*(np.sin(kx)-1j*np.sin(ky))],[a*(np.sin(kx)+1j*np.sin(ky)),-mu+t*(np.cos(kx)+np.cos(ky))]))
    
    return H

def QHZ_real_space(kx,Ny,t,mu,alpha,disorder=0,SB=0):
    if disorder==0:
        disorder=[0 for i in range(Ny)]
    
    H=np.zeros((2*Ny,2*Ny))
    
    for y in range(Ny-1):
        H[2*y,2*(y+1)]=-t/2+disorder[y]
        H[2*y+1,2*(y+1)+1]=t/2-disorder[y]
        
        H[2*(y+1),2*y+1]=alpha
        H[2*y,2*(y+1)+1]=-alpha
        
    for y in range(Ny):
        H[2*y,2*y+1]=alpha*np.sin(kx)
        
    
    H+=np.conj(H.T)
    
    
    for y in range(Ny):
        H[2*y,2*y]=mu-t*np.cos(kx)+disorder[y]
        H[2*y+1,2*y+1]=t*np.cos(kx)-mu-disorder[y]
    y=0
    H[2*y,2*y]+=SB
    H[2*y+1,2*y+1]+=SB
        
   
    return H
#Green's functions

def TB_surface_Greens_function(omega,kx,Ny,t,mu,alpha,disorder=0,SB=0,eta=0.0000001j):
    G=np.linalg.inv((omega+eta)*np.identity(2*Ny)-QHZ_real_space(kx, Ny, t, mu, alpha,disorder=disorder,SB=SB))
    
    GS=G[:2,:2]
    
    return GS

def recusive_QWZ_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j,threshold=10**(-6)):
    E=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    a=np.array(([-t/2,-alpha],[alpha,t/2]))
    b=np.conj(a.T)
    Es=E
    
    complete=False
    iterations=0
    
    while complete==False:
        Es_new=Es+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
        a_new=a@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
        b_new=b@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
        E_new=E+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b+b@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
        
        change=np.sqrt(np.sum(abs(Es_new-Es)**2))/4
        if change>=threshold:
            Es=Es_new
            a=a_new
            b=b_new
            E=E_new
            
            iterations+=1
           
            continue
        elif change<threshold:
            GF_surf=np.linalg.inv((omega+eta)*np.identity(2)-Es_new)
            complete=True
    return GF_surf

def d_sigma(kx,z,t,mu,alpha,sign_y):

    dx=alpha*np.sin(kx)*z
    dy=alpha*(z**2-1)/(2j)*sign_y
    dz=mu*z-np.cos(kx)*z-(z**2+1)/2
    
    sigma_x=np.array(([0,1],[1,0]),dtype=complex)
    sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
    sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
    return dx*sigma_x+dy*sigma_y+dz*sigma_z

def coeff_a0(alpha):
    return alpha**2/4-1/4

def coeff_a1(kx,mu):
    return (mu-np.cos(kx))

def coeff_a2(omega,kx,mu,alpha,eta=0.00001j):
    return (omega+eta)**2-1/2-(mu-np.cos(kx))**2-alpha**2*np.sin(kx)**2-alpha**2/2

def poles(omega,kx,mu,alpha,pm1,pm2,eta=0.00001j):
    

    a=coeff_a0(alpha)
    b=coeff_a1(kx, mu)
    c=coeff_a2(omega, kx, mu, alpha,eta=eta)
    
    
    pole=(-b+pm1*np.emath.sqrt(8*a**2-4*a*c+b**2))/(4*a)+pm2/2*np.emath.sqrt(b**2/(2*a**2)-c/a-2-pm1*b/(2*a**2)*np.emath.sqrt(b**2-4*a*c+8*a**2))
    return pole


def analytic_GF(omega,kx,y,t,mu,alpha,eta=0.00001j):
       
    if alpha==t:
        a1=coeff_a1(kx, mu)
        a2=coeff_a2(omega, kx, mu, alpha)
        p_plus=(-a2+np.sqrt(a2**2-4*a1**2))/(2*a1)
        p_min=(-a2-np.sqrt(a2**2-4*a1**2))/(2*a1)
        
        pole_values=[0,p_plus,p_min]
    
    else:
        pm=[1,-1]
        pole_values=np.array(([poles(omega, kx, mu, alpha, pm1, pm2,eta=eta) for pm1,pm2 in itr.product(pm,pm)]))
    g=np.zeros((2,2),dtype=complex)
    
    if y==0:
        sign_y=1
    else:
        sign_y=np.sign(y)
    
    # if mu==0:
    #     sign_mu=1
    # else:
    #     sign_mu=np.sign(mu)
    
    for p_indx,p in enumerate(pole_values):
        if abs(p)<1:
            
            rm_pole_values=np.delete(pole_values,p_indx)
            
            denominator=np.prod(p-rm_pole_values)
            
            g+=1/(t*(alpha**2/4-1/4)*denominator)*(p**(abs(y))*((omega+eta)*p*np.identity(2)+d_sigma(kx, p, t, mu, alpha,sign_y)))
 
    return g


def analytic_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha],[alpha,t/2]))
    G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
    GF_surf=np.linalg.inv(((omega+eta)*np.identity(2)-(H0@G0+H1@G1)@np.linalg.inv(G0)))
    
    return GF_surf

#Electronic Structure

def LDOS(omega,kx,t,mu,alpha):
    G=analytic_GF(omega, kx, 0, t, mu, alpha)
    
    LDOS_value=-1/np.pi*np.imag(np.trace(G))
    
    return LDOS_value

def recursive_SDOS(omega,kx,t,mu,alpha,eta=0.00001j):
    #Surface Density of States
    GS=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha,eta=eta)
    
    LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
    return LDOS_values
def TB_SDOS(omega,kx,Ny,t,mu,alpha,eta=0.00001j):
    #Surface Density of States
    GS=TB_surface_Greens_function(omega, kx, Ny, t, mu, alpha,eta=eta)
    
    LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
    return LDOS_values
def analytic_SDOS(omega,kx,t,mu,alpha,eta=0.00001j):
    #Surface Density of States
    GS=analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=eta)
    
    LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
    return LDOS_values

#Luttinger and Fermi Surfaces
def analytic_bulk_luttinger_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    GF=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    
    svd=np.min(np.linalg.svd(GF)[1])
    
    return  svd

def analytic_bulk_luttinger_surface(kx,t,mu,alpha,eta=0.0001j):
    zero_condition=lambda omega:analytic_bulk_luttinger_surface_condition(omega, kx, t, mu, alpha,eta=eta)
    
    zero_pos=fsolve(zero_condition,x0=0.1)
    zero_neg=fsolve(zero_condition,x0=-0.1)    
    return zero_pos,zero_neg


def analytic_luttinger_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    GF=analytic_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j)
    
    svd=np.min(np.linalg.svd(GF)[1])
    
    return  svd

def analytic_luttinger_surface(kx,t,mu,alpha,eta=0.000001j):
    zero_condition=lambda omega:analytic_luttinger_surface_condition(omega, kx, t, mu, alpha)
    
    zero_pos=fsolve(zero_condition,x0=0)
    
    return zero_pos

def analytic_fermi_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    GF=analytic_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j)
    
    svd=1/np.max(np.linalg.svd(GF)[1])
    return svd

def analytic_fermi_surface(kx,t,mu,alpha,eta=0.0001j):
    zero_condition=lambda omega:fermi_surface_condition(omega, kx, t, mu, alpha)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero

def luttinger_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    GF=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha)
    
    svd=np.min(np.linalg.svd(GF)[1])
    
    return  svd

def luttinger_surface(kx,t,mu,alpha,eta=0.000001j):
    zero_condition=lambda omega:luttinger_surface_condition(omega, kx, t, mu, alpha)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero

def fermi_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    GF=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha)
    
    svd=1/np.max(np.linalg.svd(GF)[1])
    return svd

def fermi_surface(kx,t,mu,alpha,eta=0.0001j):
    zero_condition=lambda omega:fermi_surface_condition(omega, kx, t, mu, alpha)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero


def disorder_luttinger_surface_condition(omega,kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
    GF=TB_surface_Greens_function(omega, kx,Ny, t, mu, alpha,disorder)
    
    svd=np.min(np.linalg.svd(GF)[1])
    
    return  svd

def disorder_luttinger_surface(kx,Ny,t,mu,alpha,disorder,eta=0.000001j):
    zero_condition=lambda omega:disorder_luttinger_surface_condition(omega, kx,Ny, t, mu, alpha,disorder)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero

def disorder_fermi_surface_condition(omega,kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
    GF=TB_surface_Greens_function(omega, kx,Ny, t, mu, alpha,disorder)
    
    svd=1/np.max(np.linalg.svd(GF)[1])
    return svd

def disorder_fermi_surface(kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
    zero_condition=lambda omega:disorder_fermi_surface_condition(omega, kx,Ny, t, mu, alpha,disorder)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero

def SB_luttinger_surface_condition(omega,kx,Ny,t,mu,alpha,SB,eta=0.0001j):
    GF=TB_surface_Greens_function(omega, kx,Ny, t, mu, alpha,SB=SB)
    
    svd=np.min(np.linalg.svd(GF)[1])
    
    return  svd

def SB_luttinger_surface(kx,Ny,t,mu,alpha,SB,eta=0.000001j):
    zero_condition=lambda omega:SB_luttinger_surface_condition(omega, kx,Ny, t, mu, alpha,SB)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero

def SB_fermi_surface_condition(omega,kx,Ny,t,mu,alpha,SB,eta=0.0001j):
    GF=TB_surface_Greens_function(omega, kx,Ny, t, mu, alpha,SB=SB)
    
    svd=1/np.max(np.linalg.svd(GF)[1])
    return svd

def SB_fermi_surface(kx,Ny,t,mu,alpha,SB,eta=0.0001j):
    zero_condition=lambda omega:SB_fermi_surface_condition(omega, kx,Ny, t, mu, alpha,SB)
    
    zero=fsolve(zero_condition,x0=0)
    
    return zero


#Topological Properties--------------------------------------------------------

def pfaffian_surface_invariant(t,mu,alpha,eta=0.000001j):
    # hs_0=np.linalg.inv(analytic_surface_Greens_function(0, 0, t, mu, alpha,eta=eta))
    # hs_pi=np.linalg.inv(analytic_surface_Greens_function(0, np.pi, t, mu, alpha,eta=eta))
    hs_0=np.linalg.inv(recusive_QWZ_surface_Greens_function(0, 0, t, mu, alpha,eta=eta))
    hs_pi=np.linalg.inv(recusive_QWZ_surface_Greens_function(0, np.pi, t, mu, alpha,eta=eta))
    
    
    
    U=1/np.sqrt(2)*np.array(([1,-1j],[1,1j]))
    
    hs_0_maj=np.round(np.conj(U.T)@hs_0@U,decimals=5)
    hs_pi_maj=np.round(np.conj(U.T)@hs_pi@U,decimals=5)
    
    try:
        invariant=np.sign(np.real(pf.pfaffian(hs_0_maj)*pf.pfaffian(hs_pi_maj)))
    except AssertionError:
        invariant=0
    
    return invariant

def pfaffian_bulk_invariant(t,mu,alpha,eta=0.000001j):
    # hs_0=np.linalg.inv(analytic_surface_Greens_function(0, 0, t, mu, alpha,eta=eta))
    # hs_pi=np.linalg.inv(analytic_surface_Greens_function(0, np.pi, t, mu, alpha,eta=eta))
    hs_0=np.linalg.inv(analytic_GF(0,0,0, t, mu, alpha,eta=eta))
    hs_pi=np.linalg.inv(np.linalg.inv(analytic_GF(0,np.pi,0, t, mu, alpha,eta=eta)))
    # hs_0=np.linalg.inv(TB_surface_Greens_function(0, 0, 101, t, mu, alpha))
    # hs_pi=np.linalg.inv(TB_surface_Greens_function(0, np.pi, 101, t, mu, alpha))
    
    hs_0_herm=(hs_0+np.conj(hs_0.T))/2
    hs_pi_herm=(hs_pi+np.conj(hs_pi.T))/2
    
    
    U=1/np.sqrt(2)*np.array(([1,-1j],[1,1j]))
    
    hs_0_maj=np.round(np.conj(U.T)@hs_0_herm@U,decimals=5)
    hs_pi_maj=np.round(np.conj(U.T)@hs_pi_herm@U,decimals=5)
    
    # try:
    invariant=np.sign(np.real(pf.pfaffian(hs_0_maj)*pf.pfaffian(hs_pi_maj)))
    # except AssertionError:
    #     invariant=0
    
    return invariant

def N1_surface_invariant(kx,t,mu,alpha):
    omega_values=np.linspace(-10,10,1001)
    domega=(omega_values[1]-omega_values[0])/2
    N1_value=0
    
    for omega in omega_values:
        N1_value+=1/(1j*np.pi)*np.trace(np.linalg.inv(analytic_surface_Greens_function(1j*omega, kx, t, mu, alpha,eta=0))@(recusive_QWZ_surface_Greens_function(1j*omega+domega, kx, t, mu, alpha,eta=0)-recusive_QWZ_surface_Greens_function(1j*omega-domega, kx, t, mu, alpha,eta=0)))
    
    return N1_value

def N1_local_invariant(kx,t,mu,alpha):
    omega_values=np.linspace(-10,10,1001)
    domega=(omega_values[1]-omega_values[0])/2
    N1_value=0
    
    for omega in omega_values:
        N1_value+=1/(1j*np.pi)*np.trace(np.linalg.inv(analytic_GF(1j*omega, kx, 0, t, mu, alpha,eta=0))@(analytic_GF(1j*omega+domega, kx, 0, t, mu, alpha,eta=0)-analytic_GF(1j*omega-domega, kx, 0, t, mu, alpha,eta=0)))
    
    return N1_value


#BHZ Model

def BHZ_d_sigma(kx,z,t,mu,alpha,sign_y):

    dx=alpha*np.sin(kx)*z
    dy=alpha*(z**2-1)/(2j)*sign_y
    dz=mu*z-t*np.cos(kx)*z-t*(z**2+1)/2
    
    iden=np.identity(2)
    sigma_x=np.array(([0,1],[1,0]),dtype=complex)
    sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
    sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
    return dx*np.kron(sigma_z,sigma_x)+dy*np.kron(iden,sigma_y)+dz*np.kron(iden,sigma_z)

def analytic_BHZ_local_Greens_function(omega,kx,y,t,mu,alpha,eta=0.0001j):
    
    pm=[1,-1]
    pole_values=np.array(([poles(omega, kx, mu, alpha, pm1, pm2,eta=eta) for pm1,pm2 in itr.product(pm,pm)]))
    
    g=np.zeros((4,4),dtype=complex)
    
    if y==0:
        sign_y=1
    else:
        sign_y=np.sign(y)
    
    # if mu==0:
    #     sign_mu=1
    # else:
    #     sign_mu=np.sign(mu)
    
    for p_indx,p in enumerate(pole_values):
        if abs(p)<1:
            
            rm_pole_values=np.delete(pole_values,p_indx)
            
            denominator=np.prod(p-rm_pole_values)
            
            g+=1/(t*(alpha**2/4-1/4)*denominator)*(p**(abs(y))*((omega+eta)*p*np.identity(4)+BHZ_d_sigma(kx, p, t, mu, alpha,sign_y)))
 
    return g

def analytic_BHZ_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.0001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx),0,0],[alpha*np.sin(kx),t*np.cos(kx)-mu,0,0],[0,0,mu-t*np.cos(kx),-alpha*np.sin(kx)],[0,0,-alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha,0,0],[alpha,t/2,0,0],[0,0,-t/2,-alpha],[0,0,alpha,t/2]))
    G0=analytic_BHZ_local_Greens_function(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_BHZ_local_Greens_function(omega, kx, 1, t, mu, alpha,eta=eta)
    
    GF_surf=np.linalg.inv(((omega+eta)*np.identity(4)-(H0@G0+H1@G1)@np.linalg.inv(G0)))
    
    return GF_surf

def analytic_SC_BHZ_surface_Greens_function(omega,kx,t,mu,alpha,Delta,gamma,eta=0.00001j):
    G_0=analytic_BHZ_surface_Greens_function(omega, kx, t, mu, alpha)
    
    H_SC=np.array(([-2*t*np.cos(kx)-mu,0,0,Delta],[0,2*t*np.cos(kx)+mu,-Delta,0],[0,-Delta,-2*t*np.cos(kx)-mu,0],[Delta,0,0,2*t*np.cos(kx)+mu]))
    
    g_SC=np.linalg.inv((omega+eta)*np.identity(4)-H_SC)
    
    T=np.array(([-gamma,0,0,0],[0,gamma,0,0],[0,0,-gamma,0],[0,0,0,gamma]))
    
    G_S=np.linalg.inv(np.linalg.inv(G_0)-T@g_SC@T)
    
    return G_S


#Induced Superconductivity tight binding models


    
    
    
