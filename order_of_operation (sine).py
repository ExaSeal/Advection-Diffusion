import numpy as np
import matplotlib.pyplot as plt
from advection_diffusion import *
from initial_conditions import *
from scipy import stats
import math as m
from misc_functions import *


# Square wave settings
A = 0.3
x_start = 0.2
x_end = 0.4
k = 2

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
K = 1e-3
u = 1
nx = 50
nt = 50
endtime = 1
steps = 10

phi = np.zeros(nx)

phi_analytic = np.zeros(nx)
dx = L / nx
dt = endtime / nt
C = u * dt / dx
D = K * dt / (dx**2)


x = np.arange(start_length, end_length, dx)
nt_list = np.unique(np.linspace(0, nt, steps, dtype=int))
print(f"Simulation initilized for T={endtime}, K ={K}, u={u}:")
print(f"nt = {nx}, nt = {nt}")
print(f"dx = {dx}, dx = {dt}")
print(f"Showing {len(nt_list)} frames")
# %% Required arrays
error_same = np.zeros(len(nt_list))
error_adv = np.zeros(len(nt_list))
error_dif = np.zeros(len(nt_list))
error_CNCS = np.zeros(len(nt_list))
error_CNCS_AD = np.zeros(len(nt_list))
error_CNCS_DA = np.zeros(len(nt_list))
error_FTBS = np.zeros(len(nt_list))
error_FTBS_AD = np.zeros(len(nt_list))
error_FTBS_DA = np.zeros(len(nt_list))
# %%


for p in range(len(nt_list)):
    plt.figure(p)
    nt = nt_list[p]

    for j in range(nx):
        x[j] = dx * j
        phi[j] = analytical_sine_adv_dif(
            u, K, k, L, A, x[j], 0
        ) + analytical_sine_adv_dif(u, K, 3, L, A, x[j], 0)

        phi_analytic[j] = analytical_sine_adv_dif(
            u, K, k, L, A, x[j], nt * dt
        ) + analytical_sine_adv_dif(u, K, 3, L, A, x[j], nt * dt)
    # Solve for Adv_Dif using different schemes and record values

    phi_same = BTCS_Adv_Dif_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_adv = BTCS_Adv1_Dif2_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_dif = BTCS_Adv2_Dif1_Periodic(phi.copy(), u, K, dx, dt, nt)
    phi_CNCS = CNCS(phi, u, K, dx, dt, nt)
    phi_CNCS_AD = CNCS_AD(phi, u, K, dx, dt, nt)
    phi_CNCS_DA = CNCS_DA(phi, u, K, dx, dt, nt)
    # phi_FTBSCS = FTBSCS_Adv_Dif_periodic(phi, u, K, dx, dt, nt)
    # phi_FTBSCS_AD = FTBSCS_Adv1_Dif2_periodic(phi, u, K, dx, dt, nt)
    # phi_FTBSCS_DA = FTBSCS_Adv2_Dif1_periodic(phi, u, K, dx, dt, nt)

    phi_schemes = [
        phi_same,
        phi_adv,
        phi_dif,
        phi_CNCS,
        phi_CNCS_AD,
        phi_CNCS_DA,
        # phi_FTBSCS,
        # phi_FTBSCS_AD,
        # phi_FTBSCS_DA,
    ]
    phi_label = [
        "BTCS",
        "BTCS_AD",
        "BTCS_DA",
        "CNCS",
        "CNCS_AD",
        "CNCS_DA",
        # "FTBS",
        # "FTBS_AD",
        # "FTBS_DA",
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
        [-A - 0.3, A + 0.3],
    )
    # Calculate RMSE error from analytical solution

    error_same[p] = RMSE(phi_same, phi_analytic)
    error_adv[p] = RMSE(phi_adv, phi_analytic)
    error_dif[p] = RMSE(phi_dif, phi_analytic)
    error_CNCS[p] = RMSE(phi_CNCS, phi_analytic)
    error_CNCS_AD[p] = RMSE(phi_CNCS_AD, phi_analytic)
    error_CNCS_DA[p] = RMSE(phi_CNCS_AD, phi_analytic)
    error_FTBS[p] = RMSE(phi_CNCS_AD, phi_analytic)
    error_FTBS_AD[p] = RMSE(phi_CNCS_AD, phi_analytic)
    error_FTBS_DA[p] = RMSE(phi_CNCS_AD, phi_analytic)


phi_error = [
    error_same,
    error_adv,
    error_dif,
    error_CNCS,
    error_CNCS_AD,
    error_CNCS_DA,
    error_FTBS,
    error_FTBS_AD,
    error_FTBS_DA,
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
    [-A - 0.3, A + 0.3],
)

plot_error(phi_error, phi_label, dx, dt, C, D)

print("BTCS AD splitting error :", np.mean(error_same - error_adv))
print("BTCS DA splitting error :", np.mean(error_same - error_dif))
print("CNCS AD splitting error :", np.mean(error_CNCS - error_CNCS_AD))
print("CNCS DA splitting error :", np.mean(error_CNCS - error_CNCS_DA))
