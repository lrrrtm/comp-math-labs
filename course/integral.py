from scipy.integrate import quad
def calculate_l():
    integral_result, _ = quad(lambda x: 1 / (1 + x ** 2) ** (1 / 3), 0.2, 0.3)
    l = 10.20638 * integral_result
    return l

print(calculate_l())
