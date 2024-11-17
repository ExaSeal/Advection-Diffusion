# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:17:35 2024

@author: Henry Yue
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 00:30:15 2024

@author: Henry Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from advection_diffusion import *
from initial_conditions import *
from scipy import stats
import math as m

steps = 20
error_same = np.zeros(steps)
error_adv = np.zeros(steps)
error_dif = np.zeros(steps)
error_FTCS = np.zeros(steps)

dx_list = np.zeros(steps)


A = 1
x_start = 0.2
x_end = 0.4

start_length = 0
end_length = 1
L = end_length - start_length


K = 0.002
u = 1

nx = 100
phi = np.zeros(nx)
x = np.zeros(nx)
phi_analytic = np.zeros(nx)
dx = (L) / (nx)
dt = 0.0020

for p in range(steps):
    plt.figure(p)
    nt = p

    for j in range(nx):
        x[j] = dx * j
        phi[j] = analytical_sine_adv_dif(u, K, L, A, x[j], 0)

        phi_analytic[j] = analytical_sine_adv_dif(u, K, L, A, x[j], nt * dt)

    fig, ax = plt.subplots()
    ax.set_xlim(start_length, end_length)  # Set the x-axis limits
    ax.set_ylim(0, A + 0.3)  # Set the y-axis limits
    plt.plot(x, phi_analytic, alpha=0.5, label="Analytic")
    plt.plot(x, phi, alpha=0.5, label="Initial Condition", color="blue")

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_FTCS = FTCS_Upwind_periodic(phi, u, K, dx, dt, nt)

    plt.plot(x, phi_same, alpha=0.5, label="Same time", color="grey")
    plt.plot(x, phi_adv, alpha=0.5, label="adv first", color="red")
    plt.plot(x, phi_dif, alpha=0.5, label="dif frist", color="blue")
    plt.plot(x, phi_FTCS, alpha=0.5, label="FTCS", color="pink")

    plt.legend()
    plt.title(f"T={nt*dt}")

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_FTCS[p] = RMSE(phi_FTCS, phi_analytic)

    dx_list[p] = dx

plt.figure(steps + 1)
plt.ylabel("RMSE")
plt.xlabel(f"Time simulated (x*{dt}) ")
plt.plot(error_same, label="Appying whole marix", color="grey")
plt.plot(error_adv, label="Appying advection first", color="red")
plt.plot(error_dif, label="Appying diffusion first", color="blue")
plt.plot(error_FTCS, label="FTCS", color="pink")

plt.legend()
plt.yscale("log")
plt.xscale("log")
