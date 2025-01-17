# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:27:07 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import itertools as itr
# from scipy.optimize import fsolve
# from random import uniform
# plt.close("all")
# plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
# plt.rc('text', usetex=True)
# plt.rcParams.update({'font.size': 30})
# def QHZ_real_space(kx,Ny,t,mu,alpha,disorder=0):
#     if disorder==0:
#         disorder=[0 for i in range(Ny)]
    
#     H=np.zeros((2*Ny,2*Ny))
    
#     for y in range(Ny-1):
#         H[2*y,2*(y+1)]=-t/2+disorder[y]
#         H[2*y+1,2*(y+1)+1]=t/2-disorder[y]
        
#         H[2*(y+1),2*y+1]=alpha
#         H[2*y,2*(y+1)+1]=-alpha
        
#     for y in range(Ny):
#         H[2*y,2*y+1]=alpha*np.sin(kx)
        
    
#     H+=np.conj(H.T)
    
    
#     for y in range(Ny):
#         H[2*y,2*y]=mu-t*np.cos(kx)+disorder[y]
#         H[2*y+1,2*y+1]=t*np.cos(kx)-mu-disorder[y]
        
   
#     return H

# def surface_Greens_function(omega,kx,Ny,t,mu,alpha,disorder):
#     G=np.linalg.inv((omega+0.0000001j)*np.identity(2*Ny)-QHZ_real_space(kx, Ny, t, mu, alpha,disorder=disorder))
    
#     GS=G[:2,:2]
    
#     return GS


# def recusive_QWZ_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j,threshold=10**(-6)):
#     E=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
#     a=np.array(([-t/2,-alpha],[alpha,t/2]))
#     b=np.conj(a.T)
#     Es=E
    
#     complete=False
#     iterations=0
    
#     while complete==False:
#         Es_new=Es+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
#         a_new=a@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
#         b_new=b@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
#         E_new=E+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b+b@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
        
#         change=np.sqrt(np.sum(abs(Es_new-Es)**2))/4
#         if change>=threshold:
#             Es=Es_new
#             a=a_new
#             b=b_new
#             E=E_new
            
#             iterations+=1
           
#             continue
#         elif change<threshold:
#             GF_surf=np.linalg.inv((omega+eta)*np.identity(2)-Es_new)
#             complete=True
#     return GF_surf

# def SDOS(omega,kx,t,mu,alpha,eta=0.00001j):
#     #Surface Density of States
#     GS=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha,eta=eta)
    
#     LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
#     return LDOS_values

# def luttinger_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
#     GF=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha)
    
#     svd=np.min(np.linalg.svd(GF)[1])
    
#     return  svd

# def luttinger_surface(kx,t,mu,alpha,eta=0.000001j):
#     zero_condition=lambda omega:luttinger_surface_condition(omega, kx, t, mu, alpha)
    
#     zero=fsolve(zero_condition,x0=0)
    
#     return zero

# def fermi_surface_condition(omega,kx,t,mu,alpha,eta=0.0001j):
#     GF=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha)
    
#     svd=1/np.max(np.linalg.svd(GF)[1])
#     return svd

# def fermi_surface(kx,t,mu,alpha,eta=0.0001j):
#     zero_condition=lambda omega:fermi_surface_condition(omega, kx, t, mu, alpha)
    
#     zero=fsolve(zero_condition,x0=0)
    
#     return zero


# def disorder_luttinger_surface_condition(omega,kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
#     GF=surface_Greens_function(omega, kx,Ny, t, mu, alpha,disorder)
    
#     svd=np.min(np.linalg.svd(GF)[1])
    
#     return  svd

# def disorder_luttinger_surface(kx,Ny,t,mu,alpha,disorder,eta=0.000001j):
#     zero_condition=lambda omega:disorder_luttinger_surface_condition(omega, kx,Ny, t, mu, alpha,disorder)
    
#     zero=fsolve(zero_condition,x0=0)
    
#     return zero

# def disorder_fermi_surface_condition(omega,kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
#     GF=surface_Greens_function(omega, kx,Ny, t, mu, alpha,disorder)
    
#     svd=1/np.max(np.linalg.svd(GF)[1])
#     return svd

# def disorder_fermi_surface(kx,Ny,t,mu,alpha,disorder,eta=0.0001j):
#     zero_condition=lambda omega:disorder_fermi_surface_condition(omega, kx,Ny, t, mu, alpha,disorder)
    
#     zero=fsolve(zero_condition,x0=0)
    
#     return zero

Ny=51
t=1
alpha=0.25
mu=0.5
disorder_scale=0.25
disorder=[disorder_scale*uniform(-1,1) for i in range(Ny)]
kx_values=np.linspace(-0.5*np.pi,0.5*np.pi,251)
omega_values=np.linspace(-1,1,251)
zero_values=np.zeros(len(kx_values))
pole_values=np.zeros(len(kx_values))

fig,axs=plt.subplots(1,2)
ax=axs[0]
for kx_indx,kx in enumerate(tqdm(kx_values)):
    zero_values[kx_indx]=luttinger_surface(kx, t, mu, alpha)
    pole_values[kx_indx]=fermi_surface(kx, t, mu, alpha)
ax.plot(kx_values/np.pi,zero_values,"bo",label="Zeros")
ax.plot(kx_values/np.pi,pole_values,"rx",label="Poles")
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.set_ylim(top=1.5,bottom=-1.5)
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\omega/t$")
ax.legend()

ax=axs[1]
for kx_indx,kx in enumerate(tqdm(kx_values)):
    zero_values[kx_indx]=disorder_luttinger_surface(kx, Ny,t, mu, alpha,disorder)
    pole_values[kx_indx]=disorder_fermi_surface(kx,Ny, t, mu, alpha,disorder)
ax.plot(kx_values/np.pi,zero_values,"bo",label="Zeros")
ax.plot(kx_values/np.pi,pole_values,"rx",label="Poles")
ax.set_ylim(top=1.5,bottom=-1.5)
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\omega/t$")
ax.legend()

TB_spectrum=np.zeros((2*Ny,len(kx_values)))
TB_spectrum_SB=np.zeros((2*Ny,len(kx_values)))

for kx_indx,kx in enumerate(tqdm(kx_values)):
    TB_spectrum[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha))
    TB_spectrum_SB[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha,disorder=disorder))

for i in range(2*Ny):
    axs[0].plot(kx_values/np.pi,TB_spectrum[i,:],"k-")
    axs[1].plot(kx_values/np.pi,TB_spectrum_SB[i,:],"k-")