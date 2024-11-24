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


steps = 20
error_same = np.zeros(steps)
error_adv = np.zeros(steps)
error_dif = np.zeros(steps)
error_parallel = np.zeros(steps)

dx_list = np.zeros(steps)


A = 1
x_start = 0.2
x_end = 0.4

start_length = 0
end_length = 1
L = end_length - start_length


K = 0.005
u = 1

nx = 50


phi = np.zeros(nx)

phi_analytic = np.zeros(nx)
dx = (L) / (nx)
dt = 2


x = np.arange(start_length, end_length, dx)

# Check FTCS stability constraints
C = u * dt / dx
D = K * dt / (dx**2)

if C <= 1 and D <= 0.5:
    print(f"FTCS stability constraints met: C = {C:.3f}, D = {D:.3f}")

else:
    print(f"FTCS stability constraints NOT met: C = {C:.3f}, D = {D:.3f}")

for p in range(steps):
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

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_parallel = BTCS_parallel_split(phi, u, K, dx, dt, nt)

    # Plot results
    plot_scheme(
        x,
        nt,
        phi,
        phi_analytic,
        phi_schemes=[
            phi_same,
            phi_adv,
            phi_dif,
            phi_parallel,
        ],
        phi_label=[
            "Whole Matrix",
            "Advection First",
            "Diffusion First",
            "Parallel Split",
        ],
    )

    # Calculate RMSE error from analytical solution

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_parallel = RMSE(phi_parallel, phi_analytic)

fig, ax = plt.subplots(figsize=(10, 6))
plt.ylabel("RMSE")
plt.xlabel("Time simulated (x*0.05)")
plt.title(f"dx = {dx}, dt = {dt}, C = {C:.3f}, D = {D:.3f}")
plt.plot(error_same, label="Appying whole marix", color="grey", alpha=0.5)
plt.plot(error_adv, label="Appying advection first", color="red", alpha=0.5)
plt.plot(error_dif, label="Appying diffusion first", color="blue", alpha=0.5)
plt.plot(error_parallel, label="parallel", alpha=0.5)

plt.legend()
plt.yscale("log")
plt.xscale("log")
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
plt.tight_layout()
plt.show()

plot_scheme_separate(
    x,
    nt,
    phi,
    phi_analytic,
    phi_schemes=[
        phi_same,
        phi_adv,
        phi_dif,
        phi_parallel,
    ],
    phi_label=[
        "Whole Matrix",
        "Advection First",
        "Diffusion First",
        "Parallel Split",
    ],
)
