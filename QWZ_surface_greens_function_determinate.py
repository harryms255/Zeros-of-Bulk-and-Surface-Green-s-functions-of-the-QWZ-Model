# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:52:38 2025

@author: Harry MullineauxSanders
"""


from QWZ_functions_file import *


t=1
alpha=0.25
mu=0
kx_values=np.linspace(-np.pi,np.pi,501)
omega_values=np.linspace(-1,1,501)

surf_det_values=np.zeros((len(omega_values),len(kx_values)))
surf_inv_det_values=np.zeros((len(omega_values),len(kx_values)))

local_det_values=np.zeros((len(omega_values),len(kx_values)))
local_inv_det_values=np.zeros((len(omega_values),len(kx_values)))


SDOS_values=np.zeros((len(omega_values),len(kx_values)))

for omega_indx,omega in enumerate(tqdm(omega_values)):
    for kx_indx,kx in enumerate(kx_values):
        GF_surf=analytic_surface_Greens_function(omega, kx, t, mu, alpha,eta=0.00000001j)
        GF_local=analytic_GF(omega, kx, 0, t, mu, alpha,eta=0.00000001j)
        
        surf_det_values[omega_indx,kx_indx]=abs(np.linalg.det(GF_surf))
        surf_inv_det_values[omega_indx,kx_indx]=abs(np.linalg.det(np.linalg.inv(GF_surf)))
        
        local_det_values[omega_indx,kx_indx]=abs(np.linalg.det(GF_local))
        local_inv_det_values[omega_indx,kx_indx]=abs(np.linalg.det(np.linalg.inv(GF_local)))
        
        
        #SDOS_values[omega_indx,kx_indx]=-1/np.pi*np.imag(np.trace(GF_surf))
fig,axs=plt.subplots(2,2)
sns.heatmap(surf_det_values,cmap="plasma",ax=axs[0,0],cbar=False,vmax=10)
sns.heatmap(surf_inv_det_values,cmap="plasma",ax=axs[0,1],cbar=False,vmax=10)

sns.heatmap(local_det_values,cmap="plasma",ax=axs[1,0],cbar=False,vmax=10)
sns.heatmap(local_inv_det_values,cmap="plasma",ax=axs[1,1],cbar=False,vmax=10)


for m in range(2):
    for n in range(2):
        axs[m,n].invert_yaxis()
        axs[m,n].set_xlabel(r"$k_x/\pi$")
        axs[m,n].set_ylabel(r"$\omega/t$")
        
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
        
        axs[m,n].set_xticks(x_ticks,labels=x_labels)
        axs[m,n].set_yticks(y_ticks,labels=y_labels)
    
axs[0,0].set_title(r"$\det(G_S)$")
axs[0,1].set_title(r"$\det(G_S^{-1})$")
axs[1,0].set_title(r"$\det(g)$")
axs[1,1].set_title(r"$\det(g^{-1})$")