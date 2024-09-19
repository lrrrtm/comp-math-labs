import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline

def f(x):
    integrand = lambda t: np.sqrt(t) * np.sin(t)
    return integrate.quad(integrand, 0, x)[0]

def compare(func, spline, lagrange):
    sp = abs(spline - func)
    lg = abs(lagrange - func)

    if sp < lg:
        return sp, lg, 'SPLINE'
    return sp, lg, 'LAGRANGE'

x_vals = np.arange(1, 3.2, 0.2)
f_vals = np.array([f(x) for x in x_vals])

print("Results for QUANC8")
for i in range(len(x_vals)):
    print(f"x = {x_vals[i]:.1f}, f(x) = {f_vals[i]}")

spline = CubicSpline(x_vals, f_vals)

lagrange_poly = np.polynomial.laguerre.Laguerre.fit(x_vals, f_vals, 10)

x_compare = np.arange(1.1, 3.1, 0.2)
f_exact = np.array([f(x) for x in x_compare])
f_spline = spline(x_compare)
f_lagrange = lagrange_poly(x_compare)

print("\nResults for SPLINE, LAGRANGE")

for i, x in enumerate(x_compare):
    comp = compare(f_exact[i], f_spline[i], f_lagrange[i])
    print(f"x = {x:.1f}: f(x) = {f_exact[i]}, spline = {f_spline[i]} ({comp[0]:.10f}), lagrange = {f_lagrange[i]} ({comp[1]:.10f}) [({comp[2]}) BETTER]")
