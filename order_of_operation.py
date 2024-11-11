# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 00:30:15 2024

@author: Henry Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from advection_diffusion import *
from scipy import stats


def Construct_Sinewave(phi,A,k,phase,x_start,x_end):
    """
    Constructs a sine wave on the array phi from x_start to x_end with wave number k

    Parameters
    ----------
    phi : Initial space to construct upon [array]
    A : Wave amplitude [float/int]
    k : Wave number
    phase : Phase
    x_start : x value in phi to start at
    x_end : x value in phi to end at

    Returns
    -------
    phi with defined wave constructed

    """
    length = x_end-x_start
    for j in range(x_start,x_end):
        phi[j] =A*np.sin((2*np.pi*k/length)*j +phase)+phi[j]
    return(phi)

def analytical_sine_adv_dif(u,k,L,A,x,t):
    phi_analytic = A*np.exp(-K*(2*np.pi/L)**2*t)*np.sin((2*np.pi/L)*(x-u*t))
    return (phi_analytic)

steps = 10
error_same = np.zeros(steps)
error_adv = np.zeros(steps)
error_dif = np.zeros(steps)
error_ada = np.zeros(steps)
error_adad = np.zeros(steps)

dx_list=np.zeros(steps)

for p in range(steps):
    plt.figure(p)

    
    A = 1
    wave_number = 1
    phase = 0

    grid_points = 50+p
    nt = 10

    start_length = 0
    end_length = 1
    L = end_length-start_length


    K=0.5
    u=0.2
    dx = (L)/(grid_points-1)
    dt = 0.005

    phi = np.zeros(grid_points)
    x = np.zeros(grid_points)
    phi_analytic = np.zeros(grid_points)
    for j in range(grid_points):
        x[j] = dx*j
        phi[j] = A*np.sin((2*np.pi/L)*(x[j]))
        phi_analytic[j] = analytical_sine_adv_dif(u, K, L, A, x[j], nt*dt)
       
        
    fig, ax = plt.subplots()
    ax.set_xlim(start_length, end_length)  # Set the x-axis limits
    ax.set_ylim(-A-0.3, A+0.3)  # Set the y-axis limits 
    plt.plot(x,phi_analytic, alpha =0.5, label='Analytic')


    phi_same = BTCS_Adv_Dif_Periodic(phi, u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi, u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi, u, K, dx, dt, nt)
    phi_ada = BTCS_Adv_Dif_Adv_Periodic(phi, u, K, dx, dt, nt)
    phi_adad = BTCS_ADAD_Periodic(phi, u, K, dx, dt, nt)

    plt.plot(x,phi_same, alpha =0.5, label='Same time', color='grey')
    plt.plot(x,phi_adv,alpha =0.5, label='adv first', color='red')
    plt.plot(x,phi_dif,alpha =0.5, label='dif frist', color='blue')
    plt.plot(x,phi_ada,alpha =0.5, label='A-D-A', color='orange')
    plt.plot(x,phi_adad,alpha =0.5, label='A-D-A-D', color='green')

    plt.legend()

    error_same[p] = np.sqrt(sum(dx*(phi_same-phi_analytic)**2))/np.sqrt(sum(dx*(phi_analytic)**2))
    error_adv[p] = np.sqrt(sum(dx*(phi_adv-phi_analytic)**2))/np.sqrt(sum(dx*(phi_analytic)**2))
    error_dif[p] = np.sqrt(sum(dx*(phi_dif-phi_analytic)**2))/np.sqrt(sum(dx*(phi_analytic)**2))
    error_ada[p] = np.sqrt(sum(dx*(phi_ada-phi_analytic)**2))/np.sqrt(sum(dx*(phi_analytic)**2))
    error_adad[p] = np.sqrt(sum(dx*(phi_adad-phi_analytic)**2))/np.sqrt(sum(dx*(phi_analytic)**2))
    dx_list[p] = dx
    
plt.figure(steps+1)
plt.ylabel('RMSE')
plt.xlabel('Time simulated (x*0.05)')
plt.plot(dx_list,error_same, label='Appying whole marix', color='grey')
plt.plot(dx_list,error_adv, label='Appying advection first', color='red')
plt.plot(dx_list,error_dif, label='Appying diffusion first', color='blue')
plt.plot(dx_list,error_ada, label= 'A-D-A', color = 'orange')
plt.plot(dx_list,error_adad, label='A-D-A-D', color = 'green')
plt.legend()
plt.yscale('log')
plt.xscale('log')
slope_same, _, _, _, _ = stats.linregress(np.log(dx_list), np.log(error_same))
slope_adv, _, _, _, _ = stats.linregress(np.log(dx_list), np.log(error_adv))
slope_dif, _, _, _, _ = stats.linregress(np.log(dx_list), np.log(error_dif))
slope_ada, _, _, _, _ = stats.linregress(np.log(dx_list), np.log(error_ada))
slope_adad, _, _, _, _ = stats.linregress(np.log(dx_list), np.log(error_adad))
