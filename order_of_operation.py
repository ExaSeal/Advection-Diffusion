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


def plot_scheme(
    x, nt, phi_initial, phi_analytical, phi_schemes, phi_label, plot_number
):
    plt.figure(plot_number)
    fig, ax = plt.subplots()
    ax.set_xlim(start_length, end_length)  # Set the x-axis limits
    ax.set_ylim(0, A + 0.3)  # Set the y-axis limits
    plt.plot(x, phi_initial, alpha=0.5, label="Initial Condition", color="blue")
    plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")

    for scheme, label in zip(phi_schemes, phi_label):
        plt.plot(x, scheme, label=label)

    plt.title(f"T={nt*dt}")
    plt.legend()


steps = 50
error_same = np.zeros(steps)
error_adv = np.zeros(steps)
error_dif = np.zeros(steps)
error_FTCS = np.zeros(steps)
error_FTCS_adv = np.zeros(steps)
error_FTCS_dif = np.zeros(steps)

dx_list = np.zeros(steps)


A = 1
x_start = 0.2
x_end = 0.4

start_length = 0
end_length = 1
L = end_length - start_length


K = 0.001
u = 1

nx = 100
phi = np.zeros(nx)

phi_analytic = np.zeros(nx)
dx = (L) / (nx)
dt = 0.0020

x = np.arange(start_length, end_length, dx)

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

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_FTCS = FTCS_Upwind_periodic(phi.copy(), u, K, dx, dt, nt)
    phi_FTCS_adv = FTCS_Adv1_Dif2_periodic(phi, u, K, dx, dt, nt)
    phi_FTCS_dif = FTCS_Adv2_Dif1_periodic(phi, u, K, dx, dt, nt)

    plot_scheme(
        x,
        nt,
        phi,
        phi_analytic,
        phi_schemes=[
            # phi_same,
            # phi_adv,
            # phi_dif,
            #  phi_FTCS,
            phi_FTCS_adv,
            phi_FTCS_dif,
        ],
        phi_label=[
            # "Whole Matrix",
            # "Advection First",
            # "Diffusion First",
            # "FTCS",
            "FTCS adv",
            "FTCS dif",
        ],
        plot_number=p,
    )

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_FTCS[p] = RMSE(phi_FTCS, phi_analytic)
    error_FTCS_adv[p] = RMSE(phi_FTCS_adv, phi_analytic)
    error_FTCS_dif[p] = RMSE(phi_FTCS_dif, phi_analytic)

    dx_list[p] = dx

plt.figure(steps + 1)
plt.ylabel("RMSE")
plt.xlabel("Time simulated (x*0.05)")
plt.plot(error_same, label="Appying whole marix", color="grey")
plt.plot(error_adv, label="Appying advection first", color="red")
plt.plot(error_dif, label="Appying diffusion first", color="blue")
plt.plot(error_FTCS, label="FTCS", color="pink")
plt.plot(
    error_FTCS_adv,
    label="FTCS adv",
)
plt.plot(
    error_FTCS_dif,
    label="FTCS dif",
)

plt.legend()
plt.yscale("log")
plt.xscale("log")
