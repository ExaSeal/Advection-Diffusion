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
    RMSE = np.sqrt(sum((phi - phi_analytic) ** 2 / len(phi)))
    return RMSE


def l_2_norm(phi, phi_analytic, dx):
    """
    Calculate the l2 normalized error for convergence test over dx

    Parameters
    ----------
    phi : Estimate
    phi_analytic : True value
    dx : Space step size

    Returns
    -------
    RMSE : Root Mean Square Error
    """
    RMSE = np.sqrt(sum(dx * (phi - phi_analytic) ** 2)) / np.sqrt(
        sum(dx * (phi) ** 2)
    )

    return RMSE


def BTCS_parallel_split(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_periodic(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for i in range(nt):
        phi_a = np.dot(np.linalg.inv(M_a), phi)
        phi_d = np.dot(np.linalg.inv(M_d), phi)

        phi = phi_a + phi_d - phi

    return phi


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
        phi = np.linalg.solve(M, phi)

    return phi


def BTCS_Adv1_Dif2_Periodic(phi, u, K, dx, dt, nt):
    M_a = matrix_BTCS_adv_periodic(phi, u, dt, dx)
    M_d = matrix_BTCS_dif_periodic(phi, K, dt, dx)

    for i in range(nt):
        phi_a = np.linalg.solve(M_a, phi)
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


def ADA(phi, u, K, dx, dt, nt):
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
        phi1 = np.linalg.solve(M_1, phi)
        phi2 = np.linalg.solve(M_2, phi1)
        phi = np.linalg.solve(M_1, phi2)
    return phi


def DAD(phi, u, K, dx, dt, nt):
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
        phi1 = np.linalg.solve(M_1, phi)
        phi2 = np.linalg.solve(M_2, phi1)
        phi = np.linalg.solve(M_1, phi2)
    return phi


def mixed_ADA_DAD(phi, u, K, dx, dt, nt):
    D = K * dt / dx**2
    C = u * dt / dx
    eta = np.sin(2 * np.pi / (1 + (C / D)))
    phi = eta * ADA(phi, u, K, dx, dt, nt) + (1 - eta) * DAD(
        phi, u, K, dx, dt, nt
    )
    return phi


def CNCS(phi, u, K, dx, dt, nt):
    C = u * dt / dx
    D = K * dt / dx**2
    M = np.zeros([len(phi), len(phi)])
    M_RHS = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M[i, i] = 4 + 4 * D
        M_RHS[i, i] = 4 - 4 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M[i, (i - 1) % len(phi)] = (-2 * D) - (C)
        M[i, (i + 1) % len(phi)] = (-2 * D) + (C)
        M_RHS[i, (i - 1) % len(phi)] = (2 * D) + (C)
        M_RHS[i, (i + 1) % len(phi)] = (2 * D) - (C)

    for i in range(nt):
        phi = np.linalg.solve(M, np.dot(M_RHS, phi))
    return phi


def CNCS_AD(phi, u, K, dx, dt, nt):
    C = u * dt / dx
    D = K * dt / dx**2
    M_a = np.zeros([len(phi), len(phi)])
    M_a_RHS = np.zeros([len(phi), len(phi)])
    M_d = np.zeros([len(phi), len(phi)])
    M_d_RHS = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_a[i, i] = 4
        M_a_RHS[i, i] = 4

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_a[i, (i - 1) % len(phi)] = -(C)
        M_a[i, (i + 1) % len(phi)] = +(C)
        M_a_RHS[i, (i - 1) % len(phi)] = +(C)
        M_a_RHS[i, (i + 1) % len(phi)] = -(C)

        # Diagonals (phi_j)
        M_d[i, i] = 4 + 4 * D
        M_d_RHS[i, i] = 4 - 4 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_d[i, (i - 1) % len(phi)] = -2 * D
        M_d[i, (i + 1) % len(phi)] = -2 * D
        M_d_RHS[i, (i - 1) % len(phi)] = 2 * D
        M_d_RHS[i, (i + 1) % len(phi)] = 2 * D
    for i in range(nt):
        phi_a = np.linalg.solve(M_a, np.dot(M_a_RHS, phi))
        phi = np.linalg.solve(M_d, np.dot(M_d_RHS, phi_a))
    return phi


def CNCS_DA(phi, u, K, dx, dt, nt):
    C = u * dt / dx
    D = K * dt / dx**2
    M_a = np.zeros([len(phi), len(phi)])
    M_a_RHS = np.zeros([len(phi), len(phi)])
    M_d = np.zeros([len(phi), len(phi)])
    M_d_RHS = np.zeros([len(phi), len(phi)])
    for i in range(len(phi)):
        # Diagonals (phi_j)
        M_a[i, i] = 4
        M_a_RHS[i, i] = 4

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_a[i, (i - 1) % len(phi)] = -(C)
        M_a[i, (i + 1) % len(phi)] = +(C)
        M_a_RHS[i, (i - 1) % len(phi)] = +(C)
        M_a_RHS[i, (i + 1) % len(phi)] = -(C)

        # Diagonals (phi_j)
        M_d[i, i] = 4 + 4 * D
        M_d_RHS[i, i] = 4 - 4 * D

        # Left and right of diagonals (phi_j-1 and phi_j+1) with periodic boundary
        M_d[i, (i - 1) % len(phi)] = -2 * D
        M_d[i, (i + 1) % len(phi)] = -2 * D
        M_d_RHS[i, (i - 1) % len(phi)] = 2 * D
        M_d_RHS[i, (i + 1) % len(phi)] = 2 * D
    for i in range(nt):
        phi_d = np.linalg.solve(M_d, np.dot(M_d_RHS, phi))
        phi = np.linalg.solve(M_a, np.dot(M_a_RHS, phi_d))
    return phi
