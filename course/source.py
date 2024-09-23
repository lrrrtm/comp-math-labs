import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

A = -5.399347635659524e-13
B = 0.9999999999991404
F = 1.0000000000015647
G = 1.9999999999986873
E = 1    # Коэффициент в уравнении теплопередачи
l = 0.9999993821632182

def ode_system(x, y, E):
    T1, T2 = y
    dT1_dx = T2
    dT2_dx = (-x * T2 + x * T1) / E
    return [dT1_dx, dT2_dx]

def shoot(p):
    # Начальные условия
    y0 = [A, p]
    sol = solve_ivp(ode_system, [0, l], y0, args=(E,), method='RK45', dense_output=True)
    return sol.sol(l)[0] - B

sol = root_scalar(shoot, bracket=[F, G], method='brentq')
print(f"Root after shoot method: {sol.root}")

p_optimal = sol.root
y0_optimal = [A, p_optimal]
sol_optimal = solve_ivp(ode_system, [0, l], y0_optimal, args=(E,), method='RK45', dense_output=True)

x_vals = np.linspace(0, l, 500)
T_vals = sol_optimal.sol(x_vals)[0]

print("Dots for plot:")
for i in range(len(T_vals)):
    print(f"x = {x_vals[i]}, T = {T_vals[i]}")

plt.plot(x_vals, T_vals, label='T(x)')
plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('График T(x) на интервале [0, l]')
plt.legend()
plt.grid()
plt.show()

delta = 0.01

params_variation = {
    'A': A + delta,
    'B': B + delta,
    'F': F + delta,
    'G': G + delta
}

print("\nErrors")
for param_name, param_value in params_variation.items():
    if param_name == 'A':
        y0_varied = [param_value, p_optimal]
    elif param_name == 'B':
        def shoot_varied(p):
            y0_varied = [A, p]
            sol_varied = solve_ivp(ode_system, [0, l], y0_varied, args=(E,), method='RK45', dense_output=True)
            return sol_varied.sol(l)[0] - param_value


        sol_varied = root_scalar(shoot_varied, bracket=[F, G], method='brentq')
        p_optimal_varied = sol_varied.root
        y0_varied = [A, p_optimal_varied]
    else:
        y0_varied = [A, p_optimal]

    sol_optimal_varied = solve_ivp(ode_system, [0, l], y0_varied, args=(E,), method='RK45', dense_output=True)

    T_vals_varied = sol_optimal_varied.sol(x_vals)[0]

    error = np.abs(T_vals_varied - T_vals)
    print(f"Error for {param_name}: {np.max(error):.10f}")
