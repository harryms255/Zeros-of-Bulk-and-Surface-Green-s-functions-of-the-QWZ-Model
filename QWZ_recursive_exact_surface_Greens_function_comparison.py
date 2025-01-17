# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:55:09 2025

@author: Harry MullineauxSanders
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools as itr
plt.close("all")
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 30})


def recusive_QWZ_surface_Greens_function(omega,kx,t,mu,alpha,eta=0.0000001j,threshold=10**(-6)):
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
        
        #change=(np.sum(abs(a_new-a)**2)+np.sum(abs(b_new-b)**2))/4
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
    return GF_surf,iterations


def QHZ_real_space(kx,Ny,t,mu,alpha):
    
    H=np.zeros((2*Ny,2*Ny))
    
    for y in range(Ny-1):
        H[2*y,2*(y+1)]=-t/2
        H[2*y+1,2*(y+1)+1]=t/2
        
        H[2*(y+1),2*y+1]=alpha
        H[2*y,2*(y+1)+1]=-alpha
        
    for y in range(Ny):
        H[2*y,2*y+1]=alpha*np.sin(kx)
        
    
    H+=np.conj(H.T)
    
    
    for y in range(Ny):
        H[2*y,2*y]=mu-t*np.cos(kx)
        H[2*y+1,2*y+1]=t*np.cos(kx)-mu
        
        
    return H

def surface_Greens_function(omega,kx,Ny,t,mu,alpha):
    G=np.linalg.inv((omega+0.0000001j)*np.identity(2*Ny)-QHZ_real_space(kx, Ny, t, mu, alpha))
    
    GS=G[:2,:2]
    
    return GS



Ny=101
t=1
alpha=1
mu=-1
kx_values=np.linspace(-np.pi,np.pi,101)

spectrum_recursive=np.zeros((2,len(kx_values)))
spectrum_exact=np.zeros((2,len(kx_values)))
omega=0

for kx_indx,kx in enumerate(tqdm(kx_values)):
    spectrum_recursive[:,kx_indx]=np.linalg.eigvalsh(recusive_QWZ_surface_Greens_function(omega, kx, t, mu, alpha)[0])
    spectrum_exact[:,kx_indx]=np.linalg.eigvalsh(surface_Greens_function(omega, kx, Ny, t, mu, alpha))

plt.figure()
for i in range(2):
    if i==0:
        plt.plot(kx_values, spectrum_recursive[i,:],"k-",label="Recursive")
        plt.plot(kx_values, spectrum_exact[i,:],"b.",label="Exact")
    else:
        plt.plot(kx_values, spectrum_recursive[i,:],"k-")
        plt.plot(kx_values, spectrum_exact[i,:],"b.")
        
#plt.ylim(top=1,bottom=-1)
plt.legend()




