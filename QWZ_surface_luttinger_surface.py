# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:22:40 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *


Ny=51
t=1
alpha=0.25
mu=1
SB=0.25
kx_values=np.linspace(-0.5*np.pi,0.5*np.pi,501)
omega_values=np.linspace(-1,1,251)
zero_values=np.zeros((len(kx_values)))
SB_zero_values=np.zeros((len(kx_values)))
pole_values=np.zeros(len(kx_values))

fig,axs=plt.subplots(1,2)
ax=axs[0]
for kx_indx,kx in enumerate(tqdm(kx_values)):
    zero_values[kx_indx]=analytic_luttinger_surface(kx, t, mu, alpha)
    pole_values[kx_indx]=analytic_fermi_surface(kx, t, mu, alpha)
ax.plot(kx_values/np.pi,zero_values,"bo",label="Zeros")

ax.plot(kx_values/np.pi,pole_values,"rx",label="Poles")
ax.set_xlim(left=min(kx_values)/np.pi,right=max(kx_values)/np.pi)
ax.set_ylim(top=1.5,bottom=-1.5)
ax.set_xlabel(r"$k_x/\pi$")
ax.set_ylabel(r"$\omega/t$")
ax.legend()

ax=axs[1]
for kx_indx,kx in enumerate(tqdm(kx_values)):
    SB_zero_values[kx_indx]=SB_luttinger_surface(kx, Ny,t, mu, alpha,SB)
    pole_values[kx_indx]=SB_fermi_surface(kx,Ny, t, mu, alpha,SB)
ax.plot(kx_values/np.pi,SB_zero_values,"bo",label="Zeros")
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
    TB_spectrum_SB[:,kx_indx]=np.linalg.eigvalsh(QHZ_real_space(kx, Ny, t, mu, alpha,SB=SB))

for i in range(2*Ny):
    axs[0].plot(kx_values/np.pi,TB_spectrum[i,:],"k-")
    axs[1].plot(kx_values/np.pi,TB_spectrum_SB[i,:],"k-")
    
    
    
    
    
fig,ax=plt.subplots()

kx=0.2*np.pi
omega_values=np.linspace(-0.3,0.3,501)

SVD_values=np.zeros((2,len(omega_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    SVD_values[:,omega_indx]=np.linalg.svd(analytic_surface_Greens_function(omega, kx, t, mu, alpha))[1]

for i in range(2):
    ax.plot(omega_values,SVD_values[i,:],"k-")
ax.set_xlabel(r"$\omega/t$")
ax.set_ylabel(r"$\Sigma_{g(\omega,k_x,y=0)}$")
ax.set_ylim(bottom=0,top=10)
ax.set_title(r"$k_x={:.2f}\pi$".format(kx/np.pi))
#ax.axhline(y=0,linewidth=3,linestyle="dashed")
ax.set_xlim(left=min(omega_values),right=max(omega_values))
