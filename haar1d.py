import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def f(x):

    """
    Исходная функция для разложения: f(x) = -1 / (x^2 + 1).

    Параметры:
        x: скаляр или массив NumPy с точками на оси x.

    Возвращает:
        Значение(я) функции f(x).
    """
    return -1.0 / (x * x + 1.0)


def compute_haar_coefficients(target_function, levels=3, interval=(0.0, 1.0)):

    """
    Вычисляет коэффициенты разложения по базису Хаара на отрезке [a,b].

    Формула из условия:
        c00 = ∫_a^b f(x) dx,
        d_{j,k} = ∫_{i2^{-j}}^{(i+0.5)2^{-j}} f(x) dx − ∫_{(i+0.5)2^{-j}}^{(i+1)2^{-j}} f(x) dx.

    Параметры:
        target_function: вызываемая функция f(x).
        levels: число уровней детализации (целое ≥ 0).
        interval: кортеж (a, b) границы интегрирования.

    Возвращает:
        (c00, d) где c00 — аппроксимирующий коэффициент,
        d — список уровней, каждый уровень — список коэффициентов d_{j,k}.
    """
    a, b = interval
    c00, _ = integrate.quad(target_function, a, b)
    detail_coefficients = []
    for level_index in range(levels):
        level_coeffs = []
        num_segments = 2 ** level_index
        segment_length = (b - a) / num_segments
        for segment_index in range(num_segments):
            left_start = a + segment_index * segment_length
            middle = left_start + segment_length / 2.0
            right_end = left_start + segment_length
            left_integral, _ = integrate.quad(target_function, left_start, middle)
            right_integral, _ = integrate.quad(target_function, middle, right_end)
            d_jk = left_integral - right_integral
            level_coeffs.append(d_jk)
        detail_coefficients.append(level_coeffs)
    return c00, detail_coefficients


def reconstruct_piecewise_average(target_function, level, interval=(0.0, 1.0)):

    """
    Строит кусочно-постоянную аппроксимацию уровня V_level как средние значения
    функции по равным подотрезкам.

    Параметры:
        target_function: функция f(x).
        level: уровень (число разбиений 2^level).
        interval: (a, b) границы.

    Возвращает:
        (xs, values), где xs — узлы разбиения (длина 2^level + 1),
        values — средние значения на каждом из 2^level подотрезков.
    """
    a, b = interval
    num_bins = 2 ** level
    bin_length = (b - a) / num_bins
    xs = np.arange(a, b + 1e-12, bin_length)
    averages = []
    for bin_index in range(num_bins):
        left = a + bin_index * bin_length
        right = left + bin_length
        integral_value, _ = integrate.quad(target_function, left, right)
        averages.append(integral_value / bin_length)
    return xs, np.array(averages)


def check_parseval(target_function, levels=3, interval=(0.0, 1.0)):

    """
    Проверяет приближённое равенство Парсеваля для разложения Хаара:
    ∫ f^2 ≈ c00^2 + Σ_j Σ_k d_{j,k}^2.

    Параметры:
        target_function: функция f(x).
        levels: число уровней, используемых в сумме по j.
        interval: (a, b) границы интегрирования.

    Возвращает:
        (integral_f_squared, partial_energy) — левая и правая части равенства.
    """
    a, b = interval
    integral_f_squared, _ = integrate.quad(lambda x: target_function(x) ** 2, a, b)
    c00, detail_coefficients = compute_haar_coefficients(target_function, levels=levels, interval=interval)
    sum_of_squares = c00 * c00
    for level_coeffs in detail_coefficients:
        for coeff in level_coeffs:
            sum_of_squares += coeff * coeff
    return integral_f_squared, sum_of_squares


def estimate_energy_error(target_function, levels=3, interval=(0.0, 1.0)):

    """
    Вычисляет модуль расхождения между ∫ f^2 и суммой энергий коэффициентов
    Хаара на первых levels уровнях.

    Параметры:
        target_function: функция f(x).
        levels: число уровней.
        interval: (a, b) границы.

    Возвращает:
        Невязку (неотрицательное число).
    """
    integral_f_squared, partial_energy = check_parseval(target_function, levels=levels, interval=interval)
    return abs(integral_f_squared - partial_energy)


def plot_reconstruction(target_function, level=3, interval=(0.0, 1.0)):

    """
    Рисует график исходной функции и её кусочно-постоянной аппроксимации V_level
    с вертикальными разделителями как на примере-изображении.

    Параметры:
        target_function: функция f(x).
        level: уровень аппроксимации (2^level столбиков).
        interval: (a, b) границы отрезка.
    """
    a, b = interval
    dense_x = np.linspace(a, b, 2001)
    dense_y = target_function(dense_x)
    step_x, step_values = reconstruct_piecewise_average(target_function, level, interval)
    plt.figure(figsize=(9, 4))
    plt.plot(dense_x, dense_y, color="red", linewidth=2.0, label="f(x)")
    plt.step(step_x, np.r_[step_values, step_values[-1]], where="post", color="royalblue", linewidth=1.8, label=f"Аппрокс. V_{level-1}")
    for k in range(2 ** level + 1):
        xk = a + (b - a) * k / (2 ** level)
        plt.axvline(xk, color="#32cd32", alpha=0.6, linewidth=1.0)
    plt.axhline(0.0, color="#888", linewidth=0.8)
    plt.xlim(a, b)
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.title("Разложение Хаара: f(x) и кусочно-постоянная аппроксимация")
    plt.show()


if __name__ == "__main__":

    levels = 3
    c00, d_coeffs = compute_haar_coefficients(f, levels=levels, interval=(0.0, 1.0))
    print(f"c00 = {c00:.6f}")
    for j, level in enumerate(d_coeffs):
        formatted = ", ".join(f"{c:.6f}" for c in level)
        print(f"d[{j}]: [{formatted}]")
    int_f2, energy = check_parseval(f, levels=levels, interval=(0.0, 1.0))
    print(f"∫ f^2 ≈ {int_f2:.6f},  c00^2 + Σ d^2 ≈ {energy:.6f}")
    eps = estimate_energy_error(f, levels=levels, interval=(0.0, 1.0))
    print(f"Оценка погрешности ε ≈ {eps:.6f}")
    plot_reconstruction(f, level=levels, interval=(0.0, 1.0))