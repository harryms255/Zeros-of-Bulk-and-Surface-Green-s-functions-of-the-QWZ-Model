# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:52:06 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import seaborn as sns
# import itertools as itr
# plt.close("all")

# def d_sigma(kx,z,t,mu,alpha,sign_y):

#     dx=alpha*np.sin(kx)
#     dy=alpha*(z-1/z)/(2j)*sign_y
#     dz=mu-np.cos(kx)-(z+1/z)/2
    
#     sigma_x=np.array(([0,1],[1,0]),dtype=complex)
#     sigma_y=np.array(([0,-1j],[1j,0]),dtype=complex)
#     sigma_z=np.array(([1,0],[0,-1]),dtype=complex)
    
#     return dx*sigma_x+dy*sigma_y+dz*sigma_z

# def coeff_a1(alpha):
#     return alpha**2/4-1/4

# def coeff_a2(kx,mu):
#     return (mu-np.cos(kx))

# def coeff_a3(omega,kx,mu,alpha,eta=0.00001j):
#     return (omega+eta)**2-1/2-(mu-np.cos(kx))**2-alpha**2*np.sin(kx)**2-alpha**2/2

# def poles(omega,kx,mu,Delta,pm1,pm2,eta=0.00001j):
    

#     a=coeff_a1(Delta)
#     b=coeff_a2(kx, mu)
#     c=coeff_a3(omega, kx, mu, Delta,eta=eta)
    
    
#     pole=(-b+pm1*np.emath.sqrt(8*a**2-4*a*c+b**2))/(4*a)+pm2/2*np.emath.sqrt(b**2/(2*a**2)-c/a-2-pm1*b/(2*a**2)*np.emath.sqrt(b**2-4*a*c+8*a**2))
#     return pole


# def analytic_GF(omega,kx,y,t,mu,alpha,eta=0.00001j):
#     pm=[1,-1]
#     pole_values=np.array(([poles(omega, kx, mu, alpha, pm1, pm2,eta=eta) for pm1,pm2 in itr.product(pm,pm)]))
#     g=np.zeros((2,2),dtype=complex)
    
#     if y==0:
#         sign_y=1
#     else:
#         sign_y=np.sign(y)
    
#     # if mu==0:
#     #     sign_mu=1
#     # else:
#     #     sign_mu=np.sign(mu)
    
#     for p_indx,p in enumerate(pole_values):
#         if abs(p)<1:
            
#             rm_pole_values=np.delete(pole_values,p_indx)
            
#             denominator=np.prod(p-rm_pole_values)
            
#             g+=1/(t*(alpha**2/4-1/4)*denominator)*(p**(abs(y)+1)*((omega+eta)*np.identity(2)+d_sigma(kx, p, t, mu, alpha,sign_y)))
 
#     return g


# def analytic_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.00001j):
#     H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
#     H1=np.array(([-t/2,-alpha],[alpha,t/2]))
#     G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
#     G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
#     GF_surf=np.linalg.inv(((omega+eta)*np.identity(2)-(H0@G0+H1@G1)@np.linalg.inv(G0)))
    
#     return GF_surf

# def SDOS(omega,kx,t,mu,alpha,eta=0.00001j):
#     #Surface Density of States
#     GS=analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=eta)
    
#     LDOS_values=-1/np.pi*np.trace(np.imag(GS))
    
#     return LDOS_values



t=1
alpha=0.25
mu=1
kx_values=np.linspace(-np.pi,np.pi,251)

omega_values=np.linspace(-1,1,251)

LDOS_values=np.zeros((len(omega_values),len(kx_values)))
iterations_values=np.zeros((len(omega_values),len(kx_values)))

fig,axs=plt.subplots(2,2,figsize=[12,8])

mu_values=[-1,-0.5,0.5,1]
for ax_indx,mu in enumerate(mu_values):
    ax=axs[(int(ax_indx/2))%2,ax_indx%2]
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        for omega_indx,omega in enumerate(omega_values):
            LDOS_values[omega_indx,kx_indx]=analytic_SDOS(omega, kx, t, mu, alpha,eta=0.001j)
            #iterations_values[omega_indx,kx_indx]=recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha,eta=0.001j)[1]
            
    sns.heatmap(LDOS_values,cmap="viridis",vmax=10,ax=ax,cbar=False)
    ax.invert_yaxis()
    # plt.figure()
    # sns.heatmap(iterations_values,cmap="plasma",vmin=0)
    
    x_ticks=[]
    x_labels=[]
    y_ticks=[]
    y_labels=[]
    
    for i in range(5):
        y_ticks.append(i/4*len(omega_values))
        y_labels.append(str(np.round(min(omega_values)+i/4*(max(omega_values)-min(omega_values)),2)))
        
    for i in range(5):
        x_ticks.append(i/4*len(kx_values))
        x_labels.append(str(np.round(np.min(kx_values)/np.pi+i/4*(max(kx_values)-min(kx_values))/np.pi,2)))
    
    ax.set_xticks(x_ticks,labels=x_labels)
    ax.set_yticks(y_ticks,labels=y_labels)
    ax.set_xlabel(r"$k_x/\pi$")
    ax.set_ylabel(r"$\omega/t$")
    ax.set_title(r"$\mu={:.1f}t$".format(mu))


