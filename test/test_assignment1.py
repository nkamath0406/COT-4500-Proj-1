import pytest
from assignment_1 import ApproximationAlgorithm, Bisection_Method, Newton_Raphson_Method, Fixed_Iterator_Point


# Test Bisection Method
def test_Bisection_Method():
    root = Bisection_Method(lambda x: x**2 - 4, 0, 3)
    assert abs(root - 2) < 1e-6  # Expect root around 2


# Test Approximation Algorithm
def test_approximation_algorithm():

    def sqrt_two_iteration(x):
        return (x / 2) + (1 / x)

    result1 = ApproximationAlgorithm(sqrt_two_iteration, x0=1.5, tol=1e-5)
    assert abs(result1 - 1.414213562) < 1e-5  # Approximate sqrt(2)

    import math

    def cosine_fixed_point(x):
        return math.cos(x)

    result2 = ApproximationAlgorithm(cosine_fixed_point, x0=1.0, tol=1e-5)
    assert abs(result2 -
               math.cos(result2)) < 1e-5  # Should converge to a fixed point


# Test Newton-Raphson Method
def test_Newton_Raphson_Method():
    root = Newton_Raphson_Method(lambda x: x**2 - 4, lambda x: 2 * x, x0=3)
    assert abs(root - 2) < 1e-6  # Expect root around 2


# Test Fixed Point Iteration
def test_Fixed_Iterator_Point():
    fixed_point = Fixed_Iterator_Point(lambda x: (x + 2)**0.5, x0=2)
    assert abs(fixed_point - 2) < 1e-6  # Expect fixed point at 2


if __name__ == "__main__":
    pytest.main()
