# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 06:02:31 2024

@author: Henry Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from advection_diffusion import *
from initial_conditions import *
from scipy import stats
import math as m
from scipy.stats import linregress


def RMSE(phi, phi_analytical, dx):
    RMSE = np.sqrt(sum((phi - phi_analytical) ** 2 / len(phi)))
    return RMSE


def plot_scheme(x, nt, phi_initial, phi_analytical, phi_schemes, phi_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylim(-A - 0.3, A + 0.3)

    plt.plot(x, phi_initial, alpha=0.5, label="Initial Condition", color="blue")
    plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")

    for scheme, label in zip(phi_schemes, phi_label):
        plt.plot(x, scheme, label=label)

    plt.title(f"T={nt*dt}")

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


def calc_slope(dx_list, error_list):
    log_dx = np.log10(dx_list)
    log_error = np.log10(error_list)

    slope, intercept, r_value, p_value, std_err = linregress(log_dx, log_error)

    return slope


# Set constants
A = 1
K = 1e-5
u = 0.005
C = 1
endtime = 10
L = 1

Search_for_grid_point = 200

# Initiate step-dependent lists
run_time = []
nx_add = 40
nx_list = []
nt_list = []
dx_list = []
dt_list = []


# Calculate for viable nt that result in C and endtime to be constant
for i in range(Search_for_grid_point):
    # Propose a value of nt base on a starting nx
    nt_add = (endtime * u * nx_add) / (C * L)

    # Calculate the corresponding dt
    dt_check = endtime / nt_add

    # Check if resulting dt lead to an interger value of nt that reaches endtime
    if endtime % dt_check == 0:
        # Candidate found, add
        nx_list.append(nx_add)
        nt_list.append(int(round(nt_add)))
        dx_list.append(L / nx_add)
        dt_list.append(dt_check)
        nx_add += 1
    else:
        # Continue search
        nx_add += 1
error_same = []
error_adv = []
error_dif = []
error_parallel = []
error_ADA = []
error_DAD = []
error_mixed = []
error_CNCS = []
error_CNCS_AD = []
error_CNCS_DA = []

for i, (nx, nt) in enumerate(zip(nx_list, nt_list)):
    dx = L / nx
    dt = endtime / nt
    x = np.arange(0, L, dx)
    phi = np.zeros(nx)
    phi_analytic = np.zeros(nx)

    for j in range(nx):
        phi[j] = analytical_sine_adv_dif(u, K, L, A, x[j], 0)
        phi_analytic[j] = analytical_sine_adv_dif(u, K, L, A, x[j], nt * dt)

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_parallel = BTCS_parallel_split(phi, u, K, dx, dt, nt)
    phi_ADA = ADA(phi.copy(), u, K, dx, dt, nt)
    phi_DAD = DAD(phi.copy(), u, K, dx, dt, nt)
    phi_mixed_ADA_DAD = mixed_ADA_DAD(phi.copy(), u, K, dx, dt, nt)
    phi_CNCS = CNCS(phi.copy(), u, K, dx, dt, nt)
    phi_CNCS_AD = CNCS_AD(phi.copy(), u, K, dx, dt, nt)
    phi_CNCS_DA = CNCS_DA(phi.copy(), u, K, dx, dt, nt)

    error_same.append(RMSE(phi_same, phi_analytic, dx))
    error_adv.append(RMSE(phi_adv, phi_analytic, dx))
    error_dif.append(RMSE(phi_dif, phi_analytic, dx))
    error_parallel.append(RMSE(phi_parallel, phi_analytic, dx))
    error_ADA.append(RMSE(phi_ADA, phi_analytic, dx))
    error_DAD.append(RMSE(phi_DAD, phi_analytic, dx))
    error_mixed.append(RMSE(phi_mixed_ADA_DAD, phi_analytic, dx))
    error_CNCS.append(RMSE(phi_CNCS, phi_analytic, dx))
    error_CNCS_AD.append(RMSE(phi_CNCS_AD, phi_analytic, dx))
    error_CNCS_DA.append(RMSE(phi_CNCS_DA, phi_analytic, dx))
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

    plot_scheme(x, nt, phi, phi_analytic, phi_schemes, phi_label)
C = u * dt / dx
D = K * dt / (dx**2)

plt.plot(dx_list, error_same, label="BTCS Same", marker="o")
plt.plot(dx_list, error_adv, label="BTCS Adv", marker="s")
plt.plot(dx_list, error_dif, label="BTCS Dif", marker="^")
plt.plot(dx_list, error_parallel, label="BTCS Parallel", marker="v")
plt.plot(dx_list, error_ADA, label="ADA", marker=">")
plt.plot(dx_list, error_DAD, label="DAD", marker="<")
plt.plot(dx_list, error_mixed, label="Mixed ADA/DAD", marker="x")
plt.plot(dx_list, error_CNCS, label="CNCS", marker="d")
plt.plot(dx_list, error_CNCS_AD, label="CNCS AD", marker="p")
plt.plot(dx_list, error_CNCS_DA, label="CNCS DA", marker="h")
# Customize the plot
plt.xscale("log")  # Optional: Use a logarithmic scale for dx
plt.yscale("log")  # Optional: Use a logarithmic scale for errors
plt.ylabel("RMSE")
plt.xlabel("Grid spacing (dx)")
plt.title(f"u = {u:.3}, C = {C:.3}, K = {K:.3}")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()


# Compute the order of convergence for each error list
schemes = {
    "BTCS Same": error_same,
    "BTCS Adv": error_adv,
    "BTCS Dif": error_dif,
    "BTCS Parallel": error_parallel,
    "ADA": error_ADA,
    "DAD": error_DAD,
    "Mixed ADA/DAD": error_mixed,
    "CNCS": error_CNCS,
    "CNCS AD": error_CNCS_AD,
    "CNCS DA": error_CNCS_DA,
}

# Print the slope (order of convergence) for each scheme
print("Order of Convergence:")
for name, error_list in schemes.items():
    slope = calc_slope(dx_list, error_list)
    print(f"{name}: Order = {slope:.2f}")
