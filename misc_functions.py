# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 04:57:10 2024

@author: Henry Yue
"""
from scipy import stats
import math as m
from scipy.stats import linregress
import numpy as np
import matplotlib.pyplot as plt


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


def plot_scheme(
    x,
    nt,
    dt,
    phi_initial,
    phi_analytical,
    phi_schemes,
    phi_label,
    dom=[],
    ran=[],
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(dom[0], dom[1])
    ax.set_ylim(ran[0], ran[1])

    plt.plot(x, phi_initial, alpha=0.5, label="Initial Condition", color="blue")
    plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")

    for scheme, label in zip(phi_schemes, phi_label):
        plt.plot(x, scheme, label=label)

    plt.title(f"T={nt*dt:.3}")

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
    plt.tight_layout()
    plt.show()


def plot_scheme_separate(
    x,
    nt,
    dt,
    phi_initial,
    phi_analytical,
    phi_schemes,
    phi_label,
    dom=[],
    ran=[],
):
    for scheme, label in zip(phi_schemes, phi_label):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(dom[0], dom[1])
        ax.set_ylim(ran[0], ran[1])

        plt.plot(
            x, phi_initial, alpha=0.5, label="Initial Condition", color="blue"
        )
        plt.plot(x, phi_analytical, alpha=0.5, label="Analytic")
        plt.plot(x, scheme, label=label)
        plt.title(f"T={nt*dt}")

        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)
        plt.tight_layout()
        plt.show()


def plot_error(phi_error, phi_label, dx, dt, C, D):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel("RMSE")
    plt.xlabel(f"Time simulated (x*{dt:.3})")
    plt.title(f"dx = {dx}, dt = {dt}, C = {C:.3f}, D = {D:.3f}")

    # Loop to plot errors and print average error
    for error_values, label in zip(phi_error, phi_label):
        plt.plot(error_values, label=label)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.tight_layout()
    plt.show()

    # Chat GPT provides:
    error_ranking = list(zip(phi_label, phi_error))

    # Sort by error (ascending)
    error_ranking_sorted = sorted(error_ranking, key=lambda x: np.mean(x[1]))

    # Print ranking
    for rank, (label, error) in enumerate(error_ranking_sorted, 1):
        print(f"{rank}. {label}: Mean Error = {np.mean(error):.3f}")


def calc_slope(dx_list, error_list):
    log_dx = np.log10(dx_list)
    log_error = np.log10(error_list)

    slope, intercept, r_value, p_value, std_err = linregress(log_dx, log_error)

    return slope


def find_max(C_values, D_values, kdx_list, A_eq):
    """
    Generate a array (i corresponds to Courant number, j corresponds to Diffusion number) of maximum amplifying factor given by the equation A_eq through iterating

    Parameters
    ----------
    C_values : Array of Courant numbers to test for (0.00001 to inf)
    D_values : Array of Diffusion numbers to test for (0.00001 to inf)
    kdx_list : k*dx values (0 to 2*pi)
    A_eq : Equation for the amplifcation factor (a function(D,C,kdx))

    Returns
    -------
    A_max : Array (C by D) of maximum amplification factor

    """
    A_max = np.zeros((len(C_values), len(D_values)))
    # Compute maximum A for each pair of (C, D)
    for i, C in enumerate(C_values):
        for j, D in enumerate(D_values):
            for kdx in kdx_list:
                max_A = -np.inf
                A = A_eq(D, C, kdx)
                if A > max_A:
                    max_A = A
            A_max[i, j] = max_A

    return A_max
