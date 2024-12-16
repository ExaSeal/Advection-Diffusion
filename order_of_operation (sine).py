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
k = 1

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
K = 1e-2
u = 1
nx = 100
nt = 50
endtime = 1
steps = 10

dx = L / nx
dt = endtime / nt
C = u * dt / dx
D = K * dt / (dx**2)


# Graph domain and range:
dom = [start_length, end_length]
ran = [-A - 0.3, A + 0.3]


x = np.arange(start_length, end_length, dx)
nt_list = np.unique(np.linspace(0, nt, steps, dtype=int))
print(f"Simulation initilized for T={endtime}, u ={u}, K={K}:")
print(f"C = {C}, D = {D}")
print(f"nt = {nx}, nt = {nt}")
print(f"dx = {dx}, dx = {dt}")
print(f"Showing {len(nt_list)} frames")

# %%


use_schemes = [
    (BTCS_Adv_Dif_Periodic, "BTCS"),
    (BTCS_Adv1_Dif2_Periodic, "BTCS AD"),
    (BTCS_Adv2_Dif1_Periodic, "BTCS DA"),
    (ADA, "BTCS ADA"),
    (DAD, "BTCS DAD"),
    (CNCS, "CNCS"),
    (CNCS_AD, "CNCS AD"),
    (CNCS_DA, "CNCS DA"),
]

errors = {phi_label: [] for scheme, phi_label in use_schemes}

# Loop for generating the x axis values and initial condition
phi = np.zeros(nx)
for j in range(nx):
    x[j] = dx * j
    phi[j] = analytical_sine_adv_dif(u, K, k, L, A, x[j], 0)


for p in range(len(nt_list)):
    plt.figure(p)
    nt = nt_list[p]
    phi_analytic = np.zeros(nx)

    # Calculate the analytcial solution for current time nt
    for j in range(nx):
        phi_analytic[j] = analytical_sine_adv_dif(u, K, k, L, A, x[j], nt * dt)

    # Run the schemes and record the numerical value in schemes[]
    schemes = []
    for phi_scheme, phi_label in use_schemes:
        phi_numerical = phi_scheme(phi, u, K, dx, dt, nt)
        schemes.append((phi_numerical, phi_label))
        errors[phi_label].append(RMSE(phi_numerical, phi_analytic))
        # Plot results
    plot_scheme(
        x,
        nt,
        dt,
        phi,
        phi_analytic,
        schemes,
        dom,
        ran,
    )

# Plot the final simulation result at endtime indivisually for each scheme
plot_scheme_separate(
    x,
    nt,
    dt,
    phi,
    phi_analytic,
    schemes,
    dom,
    ran,
)

plot_error(errors, dx, dt, C, D)
