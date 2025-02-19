import pandas as pd
import numpy as np
import pytest


def Bisection_Method(func, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find a root of a function `func` in the interval [a, b].

    Parameters:
    - func: The function for which the root is to be found.
    - a, b: The interval endpoints.
    - tol: Tolerance for stopping criteria (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - Root of the function if found, or None if not found within `max_iter`.
    """
    if func(a) * func(b) >= 0:
        raise ValueError("Function must have opposite signs at a and b.")

    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(func(c)) < tol or abs(b - a) < tol:
            return c
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return None

def approximation_algorithm(func, x0: float, tol: float = 1e-6, max_iter: int = 1000, learning_rate: float = 0.01) -> float:
    """
    A general approximation algorithm, such as gradient descent for optimization.

    Parameters:
    - func: The function to approximate or optimize.
    - x0: Initial guess.
    - tol: Tolerance for convergence (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).
    - learning_rate: Step size for updates (default: 0.1).

    Returns:
    - The approximate solution.
    """
    x = x0
    for _ in range(max_iter):
        # Improved numerical gradient using central difference method
        grad = (func(x + tol) - func(x - tol)) / (2 * tol)

        if abs(grad) < tol:  # Stop if gradient is too small
            break

        x_new = x - learning_rate * grad  # Gradient descent step

        if abs(x_new - x) < tol:  # Check for convergence
            return x_new

        x = x_new

    return x  # Return the last computed value if max_iter is reached


def Approximation_Algorithm(func, x0, tol=1e-6, max_iter=100):
    """
    A general approximation algorithm, such as gradient descent for optimization.

    Parameters:
    - func: The function to approximate or optimize.
    - x0: The initial guess.
    - tol: Tolerance for convergence (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - The approximate solution.
    """
    x = x0
    for _ in range(max_iter):
        grad = (func(x + tol) - func(x)) / tol  # Numerical gradient
        x_new = x - 0.1 * grad  # Update with learning rate 0.1
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def Newton_Raphson_Method(func, func_derivative, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method to find the root of a function.

    Parameters:
    - func: The function for which the root is to be found.
    - func_derivative: The derivative of the function.
    - x0: The initial guess.
    - tol: Tolerance for stopping criteria (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - The root of the function if found, or None if not found within `max_iter`.
    """
    x = x0
    for _ in range(max_iter):
        f_x = func(x)
        f_prime_x = func_derivative(x)
        if abs(f_prime_x) < tol:
            raise ValueError("Derivative too small; method may fail.")
        x_new = x - f_x / f_prime_x
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def Fixed_Iterator_Point(func, x0, tol=1e-6, max_iter=100):
    """
    Fixed Point Iteration method to find a fixed point of a function.

    Parameters:
    - func: The function for which the fixed point is to be found.
    - x0: The initial guess.
    - tol: Tolerance for stopping criteria (default: 1e-6).
    - max_iter: Maximum number of iterations (default: 100).

    Returns:
    - The fixed point if found, or None if not found within `max_iter`.
    """
    x = x0
    for _ in range(max_iter):
        x_new = func(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return None


def ApproximationAlgorithm(func, x0=1.5, tol=1e-5, max_iter=1000):
    """
    Generalized approximation algorithm.

    :param func: The function to iterate on.
    :param x0: Initial guess.
    :param tol: Convergence tolerance.
    :param max_iter: Maximum number of iterations.
    :return: Approximated value after convergence
    """
    iter_count = 0
    x = x0
    diff = float('inf')

    print(f"{iter_count}: {x}")

    while diff >= tol and iter_count < max_iter:
        iter_count += 1
        new_x = func(x)
        diff = abs(new_x - x)
        print(f"{iter_count}: {new_x}")
        x = new_x

    print(f"\nConvergence after {iter_count} iterations")
    return x

    