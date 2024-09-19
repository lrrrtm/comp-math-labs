import numpy as np
from scipy.integrate import solve_ivp

def system(t, y):
    x1, x2 = y
    dx1_dt = -45 * x1 + 60 * x2 + np.sin(1 + t)
    dx2_dt = 70 * x1 - 110 * x2 + np.cos(1 - t) + t + 1
    return [dx1_dt, dx2_dt]

y0 = [5, -1]
t_span = (0, 1)
t_eval = np.arange(0, 1.05, 0.05)
sol_rkf45 = solve_ivp(system, t_span, y0, method='RK45', t_eval=t_eval, rtol=1e-4)

def rk2_step(f, t, y, h):
    k1 = np.array(f(t, y))
    k2 = np.array(f(t + h, y + h * k1))
    return y + h / 2 * (k1 + k2)

def solve_rk2(f, t_span, y0, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = [y0]
    y = np.array(y0)
    for t in t_values[:-1]:
        y = rk2_step(f, t, y, h)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

t_rk2_025, sol_rk2_025 = solve_rk2(system, t_span, y0, 0.025)
t_rk2_custom, sol_rk2_custom = solve_rk2(system, t_span, y0, 0.0001)

print("Results for RKF45:")
for t, y1, y2 in zip(sol_rkf45.t, sol_rkf45.y[0], sol_rkf45.y[1]):
    print(f"t = {t:.2f}, x1 = {y1:.6f}, x2 = {y2:.6f}")

print("\nResults for RK2 (step=0.025)")
for t, (y1, y2) in zip(t_rk2_025, sol_rk2_025):
    print(f"t = {t:.2f}, x1 = {y1:.6f}, x2 = {y2:.6f}")

print("\nResults for RK2 (step=0.0001)")
for t, (y1, y2) in zip(t_rk2_custom, sol_rk2_custom):
    print(f"t = {t:.4f}, x1 = {y1:.6f}, x2 = {y2:.6f}")
