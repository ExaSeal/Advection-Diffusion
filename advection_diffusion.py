# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:48:52 2024

@author: Henry Yue
"""

import numpy as np

def BTCS_Adv_Dif_Periodic(phi,u,K,dx,dt,nt):
    D = K*dt/dx**2
    C = u*dt/dx
    
    
    M = np.zeros([len(phi),len(phi)])
    for i in range(len(phi)):
        #Diagonals
        M[i,i] = 1+2*D
        
        #Periodic boundary
        M[i,(i-1)%len(phi)]=-D-C/2
        M[i,(i+1)%len(phi)]=-D+C/2
    
    for i in range(nt):
        phi_new = np.dot(np.linalg.inv(M),phi)
        phi = phi_new.copy()
    
    return(phi)


def BTCS_Adv1_Dif2_Periodic(phi,u,K,dx,dt,nt):
    D = K*dt/dx**2
    C = u*dt/dx
    M_a = np.zeros([len(phi),len(phi)])
    M_d = np.zeros([len(phi),len(phi)])
    for i in range(len(phi)):
        #Diagonals
        M_a[i,i] = 1
        M_d[i,i] = 1+2*D
        
        #Periodic boundary
        M_a[i,(i-1)%len(phi)]=-C/2
        M_a[i,(i+1)%len(phi)]=C/2
        
        M_d[i,(i-1)%len(phi)]=-D
        M_d[i,(i+1)%len(phi)]=-D
        
    for i in range(nt):
        phi_a = np.dot(np.linalg.inv(M_a),phi)
        phi_new = np.dot(np.linalg.inv(M_d),phi_a)
        phi = phi_new.copy()
        
    return(phi)
        

def BTCS_Adv2_Dif1_Periodic(phi,u,K,dx,dt,nt):
    D = K*dt/dx**2
    C = u*dt/dx
    M_a = np.zeros([len(phi),len(phi)])
    M_d = np.zeros([len(phi),len(phi)])
    for i in range(len(phi)):
        #Diagonals
        M_a[i,i] = 1
        M_d[i,i] = 1+2*D
        
        #Periodic boundary
        M_a[i,(i-1)%len(phi)]=-C/2
        M_a[i,(i+1)%len(phi)]=C/2
        
        M_d[i,(i-1)%len(phi)]=-D
        M_d[i,(i+1)%len(phi)]=-D
        
    for i in range(nt):
        phi_d = np.dot(np.linalg.inv(M_d),phi)
        phi_new = np.dot(np.linalg.inv(M_a),phi_d)
        phi = phi_new.copy()
        
    return(phi)

def BTCS_Adv_Dif_Adv_Periodic(phi,u,K,dx,dt,nt):
    D = K*dt/dx**2
    C = u*dt/dx
    C = C/2
    M_a = np.zeros([len(phi),len(phi)])
    M_d = np.zeros([len(phi),len(phi)])
    for i in range(len(phi)):
        #Diagonals
        M_a[i,i] = 1
        M_d[i,i] = 1+2*D
        
        #Periodic boundary
        M_a[i,(i-1)%len(phi)]=-C/2
        M_a[i,(i+1)%len(phi)]=C/2
        
        M_d[i,(i-1)%len(phi)]=-D
        M_d[i,(i+1)%len(phi)]=-D
        
    for i in range(nt):
        phi_a = np.dot(np.linalg.inv(M_a),phi)
        phi_ad = np.dot(np.linalg.inv(M_d),phi_a)
        phi_new = np.dot(np.linalg.inv(M_a),phi_ad)
        phi = phi_new.copy()
        
    return(phi)

def BTCS_ADAD_Periodic(phi,u,K,dx,dt,nt):
    D = K*dt/dx**2
    D = D/2
    C = u*dt/dx
    C = C/2
    M_a = np.zeros([len(phi),len(phi)])
    M_d = np.zeros([len(phi),len(phi)])
    for i in range(len(phi)):
        #Diagonals
        M_a[i,i] = 1
        M_d[i,i] = 1+2*D
        
        #Periodic boundary
        M_a[i,(i-1)%len(phi)]=-C/2
        M_a[i,(i+1)%len(phi)]=C/2
        
        M_d[i,(i-1)%len(phi)]=-D
        M_d[i,(i+1)%len(phi)]=-D
        
    for i in range(nt):
        phi_a = np.dot(np.linalg.inv(M_a),phi)
        phi_ad = np.dot(np.linalg.inv(M_d),phi_a)
        phi_ada = np.dot(np.linalg.inv(M_a),phi_ad)
        phi_new = np.dot(np.linalg.inv(M_d),phi_ada)
        phi = phi_new.copy()
        
    return(phi)