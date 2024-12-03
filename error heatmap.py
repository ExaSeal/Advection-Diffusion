# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 20:04:15 2024

@author: Henry Yue
"""

import numpy as np
from initial_conditions import *
from advection_diffusion import *
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm

# wave settings
A = 1
k = 2

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
nx = 50
nt = 50
endtime = 1
t = endtime
u_values = np.linspace(0, 1, 50)  # Range of Courant numbers
K_values = np.linspace(0, 0.05, 50)  # Range of Diffusion numbers

phi_analytic = analytical_sine_adv_dif
dx = L / nx
dt = endtime / nt


def error_heatmap(
    phi_scheme, phi_split, phi_analytic, u_values, K_values, dx, dt, L, t, A
):
    x = np.arange(0, L, dx)
    error_numerical = np.zeros((len(u_values), len(K_values)))
    error_numerical_split = np.zeros((len(u_values), len(K_values)))

    for i, u in enumerate(u_values):
        for j, K in enumerate(K_values):
            # Initial conditions
            initcon = phi_analytic(u, K, k, L, A, x, 0)
            # Solve using the scheme
            phi_numerical = phi_scheme(initcon, u, K, dx, dt, int(t / dt))
            phi_numerical_split = phi_split(initcon, u, K, dx, dt, int(t / dt))
            phi_true = phi_analytic(u, K, k, L, A, x, t)

            error_numerical[i, j] = RMSE(phi_numerical, phi_true)
            error_numerical_split[i, j] = RMSE(phi_numerical_split, phi_true)
            error_split = error_numerical - error_numerical_split

    return error_split


def plot_heatmap(error, C_values, D_values, scheme_label):
    plt.figure(figsize=(8, 6))
    plt.imshow(
        error,
        origin="lower",
        extent=[C_values.min(), C_values.max(), D_values.min(), D_values.max()],
        aspect="auto",
        cmap="coolwarm",
        norm=CenteredNorm(),
    )
    plt.colorbar(label="Error difference (error non split - error split)")
    plt.xlabel("Courant Number (C)")
    plt.ylabel("Diffusion Number (D)")
    plt.title(f"{scheme_label}, nx = {nx}, nt={nt}, T={endtime}")
    plt.show()


# Convert u and K into Courant and Diffusion numbers
C_values = u_values * dt / dx
D_values = K_values * dt / dx**2

plot_heatmap(
    error_heatmap(
        BTCS_Adv_Dif_Periodic,
        BTCS_Adv1_Dif2_Periodic,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "BTCS AD split error",
)

plot_heatmap(
    error_heatmap(
        BTCS_Adv_Dif_Periodic,
        BTCS_Adv2_Dif1_Periodic,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "BTCS DA split error",
)

plot_heatmap(
    error_heatmap(
        CNCS,
        CNCS_AD,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "CNCS AD split error",
)

plot_heatmap(
    error_heatmap(
        CNCS,
        CNCS_DA,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "CNCS DA split error",
)

plot_heatmap(
    error_heatmap(
        FTBSCS_Adv_Dif_periodic,
        FTBSCS_Adv1_Dif2_periodic,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "FTCS Upwind AD split error",
)

plot_heatmap(
    error_heatmap(
        FTBSCS_Adv_Dif_periodic,
        FTBSCS_Adv2_Dif1_periodic,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "FTCS Upwind DA split error",
)

plot_heatmap(
    error_heatmap(
        ADA,
        DAD,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
    ),
    C_values,
    D_values,
    "ADA vs DAD split error",
)
