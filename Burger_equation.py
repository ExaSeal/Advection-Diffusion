# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 23:03:48 2024

@author: Henry Yue
"""

import numpy as np


def matrix_BTCS_dif_periodic(phi, K, dt, dx):
    """
    Constructs the solution matrix to act upon
    future values (phi^n+1) that solves the BTCS diffusion equation

    Parameters
    ----------
    phi : List of current values (phi^n) to diffuse foward
    K : Diffusion coefficent
    dt : Time step size
    dx : Space step size

    Returns
    -------
    M_d : the solution matrix

    """
    D = K * dt / dx**2
    M_d = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_d[i, i] = 1 + 2 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1 )with periodic boundary
        M_d[i, (i - 1) % len(phi)] = -D
        M_d[i, (i + 1) % len(phi)] = -D

    return M_d


def matrix_BTCS_nonlin_adv_periodic(phi, dt, dx):
    """
    Constructs the solution matrix to act upon
    future values (phi^n+1) that solves the BTCS nonlinear advection equation

    Parameters
    ----------
    phi : List of current values (phi^n) to advect
    dt : Time step size
    dx : Space step size

    Returns
    -------
    M_a : the solution matrix

    """
    M_a = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_a[i, i] = 1

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_a[i, (i - 1) % len(phi)] = -phi[i] * dt / (2 * dx)
        M_a[i, (i + 1) % len(phi)] = phi[i] * dt / (2 * dx)
    return M_a


def BTCS_nonlinear_Adv_Dif_Periodic(phi, K, dx, dt, nt):
    D = K * dt / dx**2
    M = np.zeros([len(phi), len(phi)])

    for j in range(nt):
        for i in range(len(phi)):
            # Diagonals (phi_j)
            M[i, i] = 1 + 2 * D

            # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
            M[i, (i - 1) % len(phi)] = -D - phi[i] * dt / (2 * dx)
            M[i, (i + 1) % len(phi)] = -D + phi[i] * dt / (2 * dx)

        phi = np.linalg.solve(M, phi)

    return phi


def BTCS_nonlinear_Adv1_Dif2_Periodic(phi, K, dx, dt, nt):
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for j in range(nt):
        M_a = matrix_BTCS_nonlin_adv_periodic(phi, dt, dx)
        phi_a = np.linalg.solve(M_a, phi)
        phi = np.linalg.solve(M_d, phi_a)

    return phi


def BTCS_nonlinear_Adv2_Dif1_Periodic(phi, K, dx, dt, nt):
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for j in range(nt):
        M_a = matrix_BTCS_nonlin_adv_periodic(phi, dt, dx)
        phi_d = np.linalg.solve(M_d, phi)
        phi = np.linalg.solve(M_a, phi_d)

    return phi
