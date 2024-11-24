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
from misc_functions import *


# Square wave settings
A = 1
x_start = 0.2
x_end = 0.4

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
K = 0.005
u = 0.5
nx = 100
nt = 100
endtime = 10
steps = 20

phi = np.zeros(nx)

phi_analytic = np.zeros(nx)
dx = L / nx
dt = endtime / nt


x = np.arange(start_length, end_length, dx)
nt_list = np.unique(np.linspace(0, endtime, steps, dtype=int))
print(f"Simulation initilized for T={endtime}, K ={K}, u={u}:")
print(f"nt = {nx}, nt = {nt}")
print(f"dx = {dx}, dt = {dt}")
print(f"Showing {len(nt_list)} frames")

error_same = np.zeros(len(nt_list))
error_adv = np.zeros(len(nt_list))
error_dif = np.zeros(len(nt_list))
error_FTBSCS = np.zeros(len(nt_list))
error_FTBSCS_adv = np.zeros(len(nt_list))
error_FTBSCS_dif = np.zeros(len(nt_list))
error_parallel = np.zeros(len(nt_list))
error_ADA = np.zeros(len(nt_list))
error_DAD = np.zeros(len(nt_list))
error_mixed = np.zeros(len(nt_list))
error_CNCS = np.zeros(len(nt_list))
error_CNCS_AD = np.zeros(len(nt_list))
error_CNCS_DA = np.zeros(len(nt_list))
# Check FTCS stability constraints
C = u * dt / dx
D = K * dt / (dx**2)

for p in range(len(nt_list)):
    plt.figure(p)
    nt = nt_list[p]

    for j in range(nx):
        phi[j] = analytical_square_adv_dif(
            x[j], 0.00000001, A, x_start, x_end, K, u
        )

        phi_analytic[j] = analytical_square_adv_dif(
            x[j], nt * dt + 0.00000001, A, x_start, x_end, K, u
        )
    # Solve for Adv_Dif using different schemes and record values

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_parallel = BTCS_parallel_split(phi, u, K, dx, dt, nt)
    phi_ADA = ADA(phi, u, K, dx, dt, nt)
    phi_DAD = DAD(phi, u, K, dx, dt, nt)
    phi_mixed_ADA_DAD = mixed_ADA_DAD(phi, u, K, dx, dt, nt)
    phi_CNCS = CNCS(phi, u, K, dx, dt, nt)
    phi_CNCS_AD = CNCS_AD(phi, u, K, dx, dt, nt)
    phi_CNCS_DA = CNCS_DA(phi, u, K, dx, dt, nt)

    phi_schemes = [
        phi_same,
        phi_adv,
        phi_dif,
        phi_parallel,
        phi_ADA,
        phi_DAD,
        phi_mixed_ADA_DAD,
        phi_CNCS,
        phi_CNCS_AD,
        phi_CNCS_DA,
    ]
    phi_label = [
        "Whole Matrix",
        "Advection First",
        "Diffusion First",
        "Parallel Split",
        "ADA",
        "DAD",
        "mixed ADA,DAD",
        "CNCS",
        "CNCS_AD",
        "CNCS_DA",
    ]

    # Plot results
    plot_scheme(
        x,
        nt,
        dt,
        phi,
        phi_analytic,
        phi_schemes,
        phi_label,
        [start_length, end_length],
        [0, A + 0.3],
    )
    # Calculate RMSE error from analytical solution

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_parallel[p] = RMSE(phi_parallel, phi_analytic)
    error_ADA[p] = RMSE(phi_ADA, phi_analytic)
    error_DAD[p] = RMSE(phi_DAD, phi_analytic)
    error_mixed[p] = RMSE(phi_mixed_ADA_DAD, phi_analytic)
    error_CNCS[p] = RMSE(phi_CNCS, phi_analytic)
    error_CNCS_AD[p] = RMSE(phi_CNCS_AD, phi_analytic)
    error_CNCS_DA[p] = RMSE(phi_CNCS_AD, phi_analytic)

phi_error = [
    error_same,
    error_adv,
    error_dif,
    error_parallel,
    error_ADA,
    error_DAD,
    error_mixed,
    error_CNCS,
    error_CNCS_AD,
    error_CNCS_DA,
]


plot_scheme_separate(
    x,
    nt,
    dt,
    phi,
    phi_analytic,
    phi_schemes,
    phi_label,
    [start_length, end_length],
    [0, A + 0.3],
)
plot_error(phi_error, phi_label, dx, dt, C, D)
