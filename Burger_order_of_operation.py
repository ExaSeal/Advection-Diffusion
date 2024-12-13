# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:07:59 2024

@author: Henry Yue
"""

from Burger_equation import *
from initial_conditions import *
from misc_functions import *
import matplotlib.pyplot as plt

# Square wave settings
A = 0.2
x_start = 0.2
x_end = 0.4
k = 1

# Domain size
start_length = 0
end_length = 1
L = end_length - start_length

# Scheme setting
K = 1e-3
u = 1
nx = 50
nt = 1
endtime = 1
steps = 10

phi = np.zeros(nx)
phi_analytic = np.zeros(nx)

dx = L / nx
dt = endtime / nt


x = np.arange(start_length, end_length, dx)
for j in range(nx):
    x[j] = dx * j
    phi[j] = analytical_sine_adv_dif(u, K, k, L, A, x[j], 0)+0.5

phi_BTCS = BTCS_nonlinear_Adv_Dif_Periodic(phi, K, dx, dt, nt)
phi_BTCS_AD = BTCS_nonlinear_Adv1_Dif2_Periodic(phi, K, dx, dt, nt)
phi_BTCS_DA = BTCS_nonlinear_Adv2_Dif1_Periodic(phi, K, dx, dt, nt)

phi_schemes = [phi_BTCS, phi_BTCS_AD, phi_BTCS_DA]
phi_label = ["BTCS", "BTCS AD", "BTCS DA"]

plot_scheme(
    x,
    nt,
    dt,
    phi,
    phi_analytic,
    phi_schemes,
    phi_label,
    [start_length, end_length],
    [0, A + 1],
)
