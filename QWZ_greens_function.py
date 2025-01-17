# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:49:42 2025

@author: Harry MullineauxSanders
"""
from QWZ_functions_file import *

t=1
alpha=0.5
mu=1
#mu_values=np.linspace(-4,4,21)
mu_values=[1]
kx_values=np.linspace(-np.pi,np.pi,501)
ky_values=np.linspace(-np.pi,np.pi,501)

spectrum=np.zeros((2,len(ky_values),len(kx_values)))

fig,axs=plt.subplots(1,3)

for mu in mu_values:
    ax=axs[0]
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        for ky_indx,ky in enumerate(ky_values):
            
            spectrum[:,ky_indx,kx_indx]=np.linalg.eigvalsh(QWZ_model(kx, ky, mu, t,alpha))
            
    for i in range(len(ky_values)):
        for j in range(2):
            ax.plot(kx_values,spectrum[j,i,:],"k-")

#GF spectrum

    GF_spectrum=np.zeros((2,len(kx_values)))

    ax=axs[1]
    for kx_indx,kx in enumerate(tqdm(kx_values)):
            
        GF_spectrum[:,kx_indx]=np.linalg.eigvalsh(analytic_GF(0, kx, 0, t, mu, alpha,eta=0))
            
    
    for j in range(2):
        ax.plot(kx_values,GF_spectrum[j,:],"k-")
        

#LDOS
    omega_values=np.linspace(-3,3,501)
    LDOS_values=np.zeros((len(omega_values),len(kx_values)))
    
    ax=axs[2]
    for kx_indx,kx in enumerate(tqdm(kx_values)):
        for omega_indx,omega in enumerate(omega_values):
                LDOS_values[omega_indx,kx_indx]=LDOS(omega, kx, t, mu, alpha)
                
    sns.heatmap(LDOS_values,cmap="plasma",vmax=5,ax=ax)
    ax.invert_yaxis()
    
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
    fig.suptitle(r"$\mu={:.2f}t$".format(mu))