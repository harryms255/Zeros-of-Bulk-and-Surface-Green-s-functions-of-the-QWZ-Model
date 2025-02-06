# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:41:31 2025

@author: Harry 
"""

from QWZ_functions_file import *

def recusive_QWZ_surface_Greens_function_SVD(omega,kx,t,mu,alpha,eta=0.00000001j,threshold=10**(-10)):
    E=np.array(([mu-t*np.cos(kx),alpha*np.sin(kx)],[alpha*np.sin(kx),t*np.cos(kx)-mu]))
    a=np.array(([-t/2,-alpha],[alpha,t/2]))
    b=np.conj(a.T)
    Es=E
    
    svd_values=np.zeros((2,1))
    svd_values[:,0]=np.linalg.svd(np.linalg.inv((omega+eta)*np.identity(2)-Es))[1]
    
    complete=False
    iterations=0
    
    while complete==False:
        Es_new=Es+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
        a_new=a@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
        b_new=b@np.linalg.inv((omega+eta)*np.identity(2)-E)@b
        E_new=E+a@np.linalg.inv((omega+eta)*np.identity(2)-E)@b+b@np.linalg.inv((omega+eta)*np.identity(2)-E)@a
        
        
        new_svd=np.reshape(np.linalg.svd(np.linalg.inv((omega+eta)*np.identity(2)-Es_new))[1],(2,1))
        svd_values=np.append(svd_values,new_svd,axis=1)
        
        change=np.sqrt(np.sum(abs(Es_new-Es)**2))/4
        if change>=threshold:
            Es=Es_new
            a=a_new
            b=b_new
            E=E_new
            
            iterations+=1
           
            continue
        elif change<threshold:
            #GF_surf=np.linalg.inv((omega+eta)*np.identity(2)-Es_new)
            complete=True
    return svd_values


t=1
alpha=0.25
mu_values=np.linspace(0,1,21)
kx=0*np.pi
omega=0

for mu in tqdm(mu_values):
#    omega=analytic_luttinger_surface(kx, t, mu, alpha,eta=0.0001j)[0]
    svd_values=recusive_QWZ_surface_Greens_function_SVD(omega, kx, t, mu, alpha)
    
    
    plt.figure(r"mu={:.2f}".format(mu))
    plt.plot(np.log10(svd_values[0,:]),"k-x")
    plt.plot(np.log10(svd_values[1,:]),"k-x")
    plt.xlabel("Iteration Number")
    plt.ylabel(r"$\log_{10}(\Sigma)$")
