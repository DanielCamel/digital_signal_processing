import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Функции преобразования Хаара
# -------------------------------

def haar_1d_transform(vector):
    n = len(vector)
    output = np.zeros_like(vector, dtype=float)
    step = 1 / np.sqrt(2)
    while n > 1:
        n //= 2
        for i in range(n):
            output[i] = (vector[2*i] + vector[2*i+1]) * step
            output[n + i] = (vector[2*i] - vector[2*i+1]) * step
        vector[:2*n] = output[:2*n]
    return output

def haar_1d_inverse(coeffs):
    n = 1
    step = 1 / np.sqrt(2)
    output = np.copy(coeffs)
    while n < len(coeffs):
        temp = np.copy(output)
        for i in range(n):
            output[2*i]   = (temp[i] + temp[n+i]) * step
            output[2*i+1] = (temp[i] - temp[n+i]) * step
        n *= 2
    return output

def haar_2d_transform(matrix):
    transformed = np.copy(matrix)
    rows, cols = matrix.shape
    for i in range(rows):
        transformed[i, :] = haar_1d_transform(transformed[i, :])
    for j in range(cols):
        transformed[:, j] = haar_1d_transform(transformed[:, j])
    return transformed

def haar_2d_inverse(matrix):
    restored = np.copy(matrix)
    rows, cols = matrix.shape
    for j in range(cols):
        restored[:, j] = haar_1d_inverse(restored[:, j])
    for i in range(rows):
        restored[i, :] = haar_1d_inverse(restored[i, :])
    return restored

# -------------------------------
# Исходная матрица (0=черный, 1=белый)
# -------------------------------

pattern = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 1]
], dtype=float)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(pattern, cmap='gray', vmin=0, vmax=1)
plt.title("Исходное изображение (0/1)")

# -------------------------------
# Замена 0 -> -1
# -------------------------------
pattern_centered = 2 * pattern - 1

# -------------------------------
# Прямое 2D-преобразование Хаара
# -------------------------------
coeffs = haar_2d_transform(pattern_centered)

# Нормализация коэффициентов для визуализации
coeffs_vis = (coeffs - coeffs.min()) / (coeffs.max() - coeffs.min())

plt.subplot(1, 3, 2)
plt.imshow(coeffs_vis, cmap='gray')
plt.title("Коэффициенты Хаара")

# -------------------------------
# Обратное преобразование
# -------------------------------
restored = haar_2d_inverse(coeffs)

# Возвращаем диапазон в [0, 1]
restored_binary = (restored + 1) / 2
restored_binary = np.clip(restored_binary, 0, 1)

plt.subplot(1, 3, 3)
plt.imshow(restored_binary, cmap='gray', vmin=0, vmax=1)
plt.title("Восстановленное изображение")

plt.tight_layout()
plt.show()

# -------------------------------
# Округление коэффициентов и повторное восстановление
# -------------------------------
coeffs_rounded = np.round(coeffs)
restored_from_rounded = haar_2d_inverse(coeffs_rounded)
restored_from_rounded_binary = np.clip((restored_from_rounded + 1) / 2, 0, 1)

mse = np.mean((pattern - restored_from_rounded_binary)**2)

print("Исходная матрица:\n", pattern)
print("\nКоэффициенты разложения (округленные):\n", np.round(coeffs))
print("\nВосстановленная матрица:\n", np.round(restored_from_rounded_binary, 2))
print(f"\nСреднеквадратичное отклонение: {mse:.4f}")
