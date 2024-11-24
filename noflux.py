# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:48:52 2024

@author: Henry Yue
"""

import numpy as np


def matrix_noflux_transform(M):
    M[0, :] = 0
    M[-1, :] = 0
    M[0, 1] = -1
    M[0, 2] = 1
    M[-1, -2] = 1
    M[-1, -3] = -1

    return M


def matrix_BTCS_dif_noflux(phi, K, dt, dx):
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

    return matrix_noflux_transform(M_d)


def matrix_BTCS_adv_noflux(phi, u, dt, dx):
    """
    Constructs the solution matrix to act upon
    future values (phi^n+1) that solves the BTCS advection equation

    Parameters
    ----------
    phi : List of current values (phi^n) to advect
    u : Advection velocity
    dt : Time step size
    dx : Space step size

    Returns
    -------
    M_a : the solution matrix

    """
    C = u * dt / dx
    M_a = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_a[i, i] = 1

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_a[i, (i - 1) % len(phi)] = -C / 2
        M_a[i, (i + 1) % len(phi)] = C / 2

    return matrix_noflux_transform(M_a)


def BTCS_parallel_split_noflux(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_noflux(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_noflux(phi, K, dt, dx)
    phi_old = phi.copy()

    for i in range(nt):
        phi_old[0] = 0
        phi_old[-1] = 0
        phi_a = np.dot(np.linalg.inv(M_a), phi_old)
        phi_d = np.dot(np.linalg.inv(M_d), phi_old)

        phi = phi_a + phi_d - phi
        phi_old = phi.copy()

    return phi


def BTCS_Adv_Dif_noflux(phi, u, K, dx, dt, nt):
    """
    Peform BTCS Adv-Dif on phi with periodic
    boundary conditions

    Parameters
    ----------
    phi : Values to adv-dif
    u : Advection velocity
    K : Diffusion coefficent
    dx : Space step size
    dt : Time step size
    nt : Number of time steps

    Returns
    -------
    phi : BTCS Adv-Dif result

    """
    D = K * dt / dx**2
    C = u * dt / dx

    M = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M[i, i] = 1 + 2 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M[i, (i - 1) % len(phi)] = -D - C / 2
        M[i, (i + 1) % len(phi)] = -D + C / 2

    M = matrix_noflux_transform(M)

    for i in range(nt):
        phi[0] = 0
        phi[-1] = 0
        phi = np.dot(np.linalg.inv(M), phi)

    return phi


def BTCS_Adv1_Dif2_noflux(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_noflux(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_noflux(phi, K, dt, dx)

    for i in range(nt):
        phi[0] = 0
        phi[-1] = 0
        phi_a = np.linalg.solve(M_a, phi)
        phi_a[0] = 0
        phi_a[-1] = 0
        phi = np.dot(np.linalg.inv(M_d), phi_a)
    return phi


def BTCS_Adv2_Dif1_noflux(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_noflux(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_noflux(phi, K, dt, dx)

    for i in range(nt):
        phi[0] = 0
        phi[-1] = 0
        phi_d = np.dot(np.linalg.inv(M_d), phi)
        phi_d[0] = 0
        phi_d[-1] = 0
        phi = np.dot(np.linalg.inv(M_a), phi_d)
    return phi


def BTCS_ADA_noflux(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx
    M_1 = np.zeros([len(phi), len(phi)])
    M_2 = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_1[i, i] = 1

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_1[i, (i - 1) % len(phi)] = -C / 4
        M_1[i, (i + 1) % len(phi)] = C / 4
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_2[i, i] = 1 + 2 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1 )with periodic boundary
        M_2[i, (i - 1) % len(phi)] = -D
        M_2[i, (i + 1) % len(phi)] = -D
    for i in range(nt):
        phi[0] = 0
        phi[-1] = 0
        phi1 = np.linalg.solve(matrix_noflux_transform(M_1), phi)
        phi1[0] = 0
        phi1[-1] = 0
        phi2 = np.linalg.solve(matrix_noflux_transform(M_2), phi1)
        phi2[0] = 0
        phi2[-1] = 0
        phi = np.linalg.solve(matrix_noflux_transform(M_1), phi2)
    return phi


def BTCS_DAD_noflux(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx
    M_1 = np.zeros([len(phi), len(phi)])
    M_2 = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_1[i, i] = 1 + D

        # Left and right of diagonals (phi_j-1 and phi_j+1 )with periodic boundary
        M_1[i, (i - 1) % len(phi)] = -D / 2
        M_1[i, (i + 1) % len(phi)] = -D / 2
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_2[i, i] = 1

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_2[i, (i - 1) % len(phi)] = -C / 2
        M_2[i, (i + 1) % len(phi)] = C / 2
    for i in range(nt):
        phi[0] = 0
        phi[-1] = 0
        phi1 = np.linalg.solve(matrix_noflux_transform(M_1), phi)
        phi1[0] = 0
        phi1[-1] = 0
        phi2 = np.linalg.solve(matrix_noflux_transform(M_2), phi1)
        phi2[0] = 0
        phi2[-1] = 0
        phi = np.linalg.solve(matrix_noflux_transform(M_1), phi2)
    return phi


def mixed_ADA_DAD_noflux(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx
    eta = np.sin(2 * np.pi / (C / D))
    phi = eta * BTCS_ADA_noflux(phi, u, K, dx, dt, nt) + (
        1 - eta
    ) * BTCS_DAD_noflux(phi, u, K, dx, dt, nt)
    return phi
