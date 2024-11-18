# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:48:52 2024

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


def matrix_BTCS_adv_periodic(phi, u, dt, dx):
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
    return M_a


def RMSE(phi, phi_analytic):
    """
    Calculate the RMSE between the
    estimate (phi) and true (phi_analytic) values

    Parameters
    ----------
    phi : Estimate
    phi_analytic : True value

    Returns
    -------
    RMSE : Root Mean Square Error
    """
    RMSE = np.sqrt(sum((phi - phi_analytic) ** 2)) / len(phi)

    return RMSE


def BTCS_Adv_Dif_Periodic(phi, u, K, dx, dt, nt):
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

    for i in range(nt):
        phi_new = np.dot(np.linalg.inv(M), phi)
        phi = phi_new.copy()

    return phi


def BTCS_Adv1_Dif2_Periodic(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_periodic(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for i in range(nt):
        phi_a = np.dot(np.linalg.inv(M_a), phi)
        phi = np.dot(np.linalg.inv(M_d), phi_a)
    return phi


def BTCS_Adv2_Dif1_Periodic(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_periodic(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for i in range(nt):
        phi_d = np.dot(np.linalg.inv(M_d), phi)
        phi = np.dot(np.linalg.inv(M_a), phi_d)
    return phi


def FTBSCS_Adv_Dif_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx

    nx = len(phi)
    phi_new = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_new[j] = (
                D * (phi[(j + 1) % nx] + phi[(j - 1) % nx] - 2 * phi[j])
                - C * (phi[(j)] - phi[(j - 1) % nx])
                + phi[j]
            )
        phi = phi_new.copy()
    return phi


def FTBSCS_Adv1_Dif2_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx

    nx = len(phi)
    phi_a = np.zeros(nx)
    phi_ad = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_a[j] = -C * (phi[(j)] - phi[(j - 1) % nx]) + phi[j]

            phi_ad[j] = (
                D * (phi_a[(j + 1) % nx] + phi_a[(j - 1) % nx] - 2 * phi_a[j])
                + phi_a[j]
            )
        phi = phi_ad.copy()
    return phi


def FTBSCS_Adv2_Dif1_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx

    nx = len(phi)
    phi_d = np.zeros(nx)
    phi_da = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_d[j] = (
                D * (phi[(j + 1) % nx] + phi[(j - 1) % nx] - 2 * phi[j])
                + phi[j]
            )

            phi_da[j] = -C * (phi_d[(j)] - phi_d[(j - 1) % nx]) + phi_d[j]

        phi = phi_da.copy()
    return phi


def FTCSCS_Adv_Dif_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / (dx**2)
    C = u * dt / dx
    nx = len(phi)
    phi_new = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_new[j] = D * (
                phi[(j + 1) % nx] + phi[(j - 1) % nx] - 2 * phi[j]
            ) - 0.5 * C * (phi[(j + 1) % nx] - phi[(j - 1) % nx] + phi[j])
        phi = phi_new.copy()
    return phi


def FTCSCS_Adv1_Dif2_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx

    nx = len(phi)
    phi_a = np.zeros(nx)
    phi_ad = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_a[j] = (
                -0.5 * C * (phi[(j + 1) % nx] - phi[(j - 1) % nx] + phi[j])
            )

            phi_ad[j] = (
                D * (phi[(j + 1) % nx] + phi[(j - 1) % nx] - 2 * phi[j])
                + phi_ad[j]
            )

        phi = phi_ad.copy()
    return phi


def FTCSCS_Adv2_Dif1_periodic(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx

    nx = len(phi)
    phi_d = np.zeros(nx)
    phi_da = np.zeros(nx)

    for i in range(nt):
        for j in range(nx):
            phi_d[j] = (
                D * (phi[(j + 1) % nx] + phi[(j - 1) % nx] - 2 * phi[j])
                + phi[j]
            )

            phi_da[j] = (
                -0.5
                * C
                * (phi_d[(j + 1) % nx] - phi_d[(j - 1) % nx] + phi_d[j])
            )

        phi = phi_da.copy()
    return phi
