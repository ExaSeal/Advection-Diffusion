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
import os

# wave settings
A = 1
k = 1

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
nx = 10
nt = 10
endtime = 2
t = endtime
u_values = np.linspace(0, 1, 20)  # Range of Courant numbers
K_values = np.linspace(0, 0.1, 20)  # Range of Diffusion numbers

phi_analytic = analytical_sine_adv_dif
dx = L / nx
dt = endtime / nt


def error_heatmap(
    phi_scheme, phi_analytic, u_values, K_values, dx, dt, L, t, A, scheme_label
):
    x = np.arange(0, L, dx)
    error_numerical = np.zeros((len(u_values), len(K_values)))

    for i, u in enumerate(u_values):
        for j, K in enumerate(K_values):
            # Initial conditions
            initcon = phi_analytic(u, K, k, L, A, x, 0)
            # Solve using the scheme
            phi_numerical = phi_scheme(initcon, u, K, dx, dt, nt)
            phi_true = phi_analytic(u, K, k, L, A, x, t)

            # Calculate RMSE
            error_numerical[i, j] = np.average(phi_true - phi_numerical)

    return error_numerical


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
    plt.colorbar(label="RMSE")
    plt.xlabel("Courant Number (C)")
    plt.ylabel("Diffusion Number (D)")
    plt.title(f"{scheme_label}, nx = {nx}, nt={nt}, T={endtime}")
    plt.show()


# Convert u and K into Courant and Diffusion numbers
C_values = u_values * dt / dx
D_values = K_values * dt / dx**2

# Generate heatmaps and save plots for each scheme
schemes = [
    (BTCS_Adv_Dif_Periodic, "BTCS"),
    (BTCS_Adv2_Dif1_Periodic, "BTCS DA"),
    (CNCS, "CNCS"),
    (CNCS_DA, "CNCS DA"),
    (ADA, "ADA"),
    (DAD, "DAD"),
]

for scheme, label in schemes:
    error = error_heatmap(
        scheme,
        analytical_sine_adv_dif,
        u_values,
        K_values,
        dx,
        dt,
        L,
        t,
        A,
        label,
    )
    plot_heatmap(error, C_values, D_values, f"{label} error")
