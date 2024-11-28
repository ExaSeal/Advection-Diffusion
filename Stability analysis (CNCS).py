# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:28:40 2024

@author: Henry Yue
"""
import numpy as np
import matplotlib.pyplot as plt
from misc_functions import *
from scipy.stats import linregress


def A(D, C, kdx):
    num = (
        1
        + ((D**2) * (1 - np.cos(kdx)) ** 2)
        - (2 * D * (1 - np.cos(kdx)) + ((C**2 / 4) * np.sin(kdx) ** 2))
    )
    den = (
        1
        + ((D**2) * (1 - np.cos(kdx)) ** 2)
        + (2 * D * (1 - np.cos(kdx)) - ((C**2 / 4) * np.sin(kdx) ** 2))
    )

    A = num / den
    return A


C_values = np.linspace(0.1, 5, 50)  # Range of Courant numbers
D_values = np.linspace(0.1, 5, 50)  # Range of Diffusion numbers
kdx_list = np.linspace(0, 2 * np.pi, 500)

A_max = find_max(C_values, D_values, kdx_list, A)

C_border = np.zeros(len(C_values))
D_border = np.zeros(len(D_values))

for j in range(len(D_values)):
    for i in range(len(C_values)):
        if A_max[i, j] > 1:
            C_border[j] = C_values[i - 1]
            D_border[j] = D_values[j]
            break

# Check if boundary exists
if C_border[0] != C_border[1]:
    slope, intercept, r_value, p_value, std_err = linregress(C_border, D_border)
    # Generate best-fit line
    C_fit = np.linspace(
        C_values.min(), C_values.max(), 100
    )  # Independent variable
    D_fit = slope * C_fit + intercept  # Dependent variable
    # Overlay best-fit line
    plt.plot(
        C_fit,
        D_fit,
        color="blue",
        linestyle="--",
        label=f"Stability border line: C = {slope:.2f}D + {intercept:.2f}",
        linewidth=2,
    )
    print(
        f"The relation between C and D is: D = {slope:.4f} * C + {intercept:.4f}"
    )


# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    A_max.T,
    origin="lower",
    extent=[C_values.min(), C_values.max(), D_values.min(), D_values.max()],
    aspect="auto",
    cmap="viridis",
    vmax=2,
)
plt.colorbar(label="Maximum A")
plt.xlabel("Courant Number (C)")
plt.ylabel("Diffusion Number (D)")


plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
