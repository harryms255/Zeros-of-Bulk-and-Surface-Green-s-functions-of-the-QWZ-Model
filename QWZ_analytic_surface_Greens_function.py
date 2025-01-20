# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:52:06 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *


t=1
alpha=0.25
mu=1.5
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


