# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:23:42 2024

@author: Henry Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from noflux import *
from advection_diffusion import *
from initial_conditions import *
from scipy import stats
import math as m


def plot_scheme(x, nt, phi_initial, phi_analytical, phi_schemes, phi_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(start_length, end_length)
    ax.set_ylim(0, A + 0.3)

    plt.plot(x, phi_initial, alpha=0.5, label="Initial Condition", color="blue")
    plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")

    for scheme, label in zip(phi_schemes, phi_label):
        plt.plot(x, scheme, label=label)

    plt.title(f"T={nt*dt}")

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


def plot_scheme_separate(
    x, nt, phi_initial, phi_analytical, phi_schemes, phi_label
):
    for scheme, label in zip(phi_schemes, phi_label):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(start_length, end_length)
        ax.set_ylim(0, A + 0.3)

        plt.plot(
            x, phi_initial, alpha=0.5, label="Initial Condition", color="blue"
        )
        plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")
        plt.plot(x, scheme, label=label)
        plt.title(f"T={nt*dt}")

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        plt.tight_layout()
        plt.show()


def plot_error(phi_error, phi_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel("RMSE")
    plt.xlabel(f"Time simulated (x*{dt})")
    plt.title(f"dx = {dx}, dt = {dt}, C = {C:.3f}, D = {D:.3f}")
    error = {}

    for error, label in zip(phi_error, phi_label):
        plt.plot(error, label=label)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


steps = 20
error_same = np.zeros(steps)
error_adv = np.zeros(steps)
error_dif = np.zeros(steps)
error_FTBSCS = np.zeros(steps)
error_FTBSCS_adv = np.zeros(steps)
error_FTBSCS_dif = np.zeros(steps)
error_parallel = np.zeros(steps)
error_ADA = np.zeros(steps)
error_DAD = np.zeros(steps)
error_mixed = np.zeros(steps)


dx_list = np.zeros(steps)


A = 1
x_start = 40
x_end = 60

start_length = 0
end_length = 100
L = end_length - start_length


K = 10
u = 1

nx = 100


phi = np.zeros(nx)

phi_analytic = np.zeros(nx)
dx = (L) / (nx)
dt = 1

x = np.arange(start_length, end_length, dx)

# Check FTCS stability constraints
C = u * dt / dx
D = K * dt / (dx**2)

if C <= 1 and D <= 0.5:
    print(f"FTCS stability constraints met: C = {C:.3f}, D = {D:.3f}")

else:
    print(f"FTCS stability constraints NOT met: C = {C:.3f}, D = {D:.3f}")

for p in range(1, steps):
    plt.figure(p)
    nt = p

    for j in range(nx):
        phi[j] = analytical_square_adv_dif(
            x[j], 0.00000001, A, x_start, x_end, K, u
        )

        phi_analytic[j] = analytical_square_adv_dif(
            x[j], nt * dt + 0.00000001, A, x_start, x_end, K, u
        )
    # Solve for Adv_Dif using different schemes and record values

    phi_same = BTCS_Adv_Dif_noflux(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_noflux(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_noflux(phi.copy(), u, K, dx, dt, nt)
    phi_parallel = BTCS_parallel_split_noflux(phi, u, K, dx, dt, nt)
    phi_ADA = BTCS_ADA_noflux(phi, u, K, dx, dt, nt)
    phi_DAD = BTCS_DAD_noflux(phi, u, K, dx, dt, nt)
    phi_mixed_ADA_DAD = mixed_ADA_DAD_noflux(phi, u, K, dx, dt, nt)

    phi_schemes = [
        phi_same,
        phi_adv,
        phi_dif,
        phi_parallel,
        phi_ADA,
        phi_DAD,
        phi_mixed_ADA_DAD,
    ]
    phi_label = [
        "Whole Matrix",
        "Advection First",
        "Diffusion First",
        "Parallel Split",
        "ADA",
        "DAD",
        "mixed ADA,DAD",
    ]

    # Plot results
    plot_scheme(x, nt, phi, phi_analytic, phi_schemes, phi_label)
    # Calculate RMSE error from analytical solution

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_parallel[p] = RMSE(phi_parallel, phi_analytic)
    error_ADA[p] = RMSE(phi_ADA, phi_analytic)
    error_DAD[p] = RMSE(phi_DAD, phi_analytic)
    error_mixed[p] = RMSE(phi_mixed_ADA_DAD, phi_analytic)

phi_error = [
    error_same,
    error_adv,
    error_dif,
    error_parallel,
    error_ADA,
    error_DAD,
    error_mixed,
]

plot_error(phi_error, phi_label)

plot_scheme_separate(x, nt, phi, phi_analytic, phi_schemes, phi_label)
