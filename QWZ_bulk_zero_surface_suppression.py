# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:12:00 2025

@author: Harry MullineauxSanders
"""

from QWZ_functions_file import *

def GH(omega,kx,t,mu,alpha,eta=0.0001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha],[alpha,t/2]))
    G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
    return (H0@G0+H1@G1)

def GHG(omega,kx,t,mu,alpha,eta=0.0001j):
    H0=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    H1=np.array(([-t/2,-alpha],[alpha,t/2]))
    G0=analytic_GF(omega, kx, 0, t, mu, alpha,eta=eta)
    G1=analytic_GF(omega, kx, 1, t, mu, alpha,eta=eta)
    
    return (H0@G0+H1@G1)@np.linalg.inv(G0)

def interferance_zero_condition(omega,kx,t,mu,alpha,eta=0.0001j):
    svd=np.linalg.svd(GH(omega, kx, t, mu, alpha))[1]
    
    condition=np.min(svd)
    
    return condition

def interferance_zero_band(kx,t,mu,alpha,eta=0.0001j):
    zero_condition=lambda omega: interferance_zero_condition(omega, kx, t, mu, alpha)
    try:
        zero_1=fsolve(zero_condition,x0=0.1)
    except LinAlgError:
        zeros_1=0
    try:
        zero_2=fsolve(zero_condition,x0=-0.1)
    except LinAlgError:
        zeros_2=0
    return zero_1,zero_2


Ny=50
t=1
alpha=0.25
mu=1
kx=0.1*np.pi
omega_values=np.linspace(-alpha,alpha,1001)
kx_values=np.linspace(-0.1*np.pi,0.1*np.pi,101)

fig,axs=plt.subplots(2,2)

SVD_values=np.zeros((8,len(omega_values)))

# for omega_indx,omega in enumerate(tqdm(omega_values)):
#     SVD_values[:2,omega_indx]=np.linalg.svd(analytic_GF(omega, kx, 0, t, mu, alpha))[1]
#     SVD_values[2:4,omega_indx]=np.linalg.svd(GH(omega, kx, t, mu, alpha))[1]
#     SVD_values[4:6,omega_indx]=np.linalg.svd(GHG(omega, kx, t, mu, alpha))[1]
#     SVD_values[6:8,omega_indx]=np.linalg.svd(analytic_surface_Greens_function(omega, kx, t, mu, alpha))[1]
    

# for i in range(4):
#     ax=axs[int(i/2),i%2]
#     for j in range(2):
#         ax.plot(omega_values,SVD_values[2*i+j,:],"k-")
#     ax.set_xlabel(r"$\omega/t$")
#     ax.set_ylabel(r"$\Sigma/t$")
#     ax.set_ylim(bottom=0,top=5)
#     ax.set_xlim(left=min(omega_values),right=max(omega_values))
    
# axs[0,0].set_title(r"$g(\omega,{:.1f}\pi,0)$".format(kx/np.pi))
# axs[0,1].set_title(r"$H_0g(\omega,{:.1f}\pi,0)\\+H_1g(\omega,{:.1f}\pi,1)$".format(kx/np.pi,kx/np.pi))
# axs[1,0].set_title(r"$[H_0g(\omega,{:.1f}\pi,0)\\+H_1g(\omega,{:.1f}\pi,1)]g^{{-1}}(\omega,{:.1f}\pi,0)$".format(kx/np.pi,kx/np.pi,kx/np.pi))
# axs[1,1].set_title(r"$g_S(\omega,{:.1f}\pi)$".format(kx/np.pi))


plt.figure()
interferance_zero_value=np.zeros((2,len(kx_values)))
spectrum=np.zeros((2*Ny,len(kx_values)))
for kx_indx, kx in enumerate(tqdm(kx_values)):
    interferance_zero_value[0,kx_indx], interferance_zero_value[1,kx_indx]=interferance_zero_band(kx, t, mu, alpha)
    spectrum[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha))
  
for i in range(2*Ny):
    plt.plot(kx_values/np.pi,spectrum[i,:],"k-")
plt.plot(kx_values/np.pi,interferance_zero_value[0,:],"r--",linewidth=3)  
plt.plot(kx_values/np.pi,interferance_zero_value[1,:],"r--",linewidth=3)  
plt.xlabel(r"$k_x/\pi$")
plt.ylabel(r"$\omega/t$")
plt.ylim(top=alpha,bottom=-alpha)
