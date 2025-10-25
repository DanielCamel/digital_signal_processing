import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os

def f12(x):
    """Функция 12: f(x) = cos(3x^4 - 3x^3 - 3x^2 + 3x)"""
    return np.cos(3*x**4 - 3*x**3 - 3*x**2 + 3*x)

def convolution_time_domain(x, h):
    """Свертка во временной области"""
    N = len(x)
    M = len(h)
    result = np.zeros(N + M - 1)
    
    for n in range(len(result)):
        for k in range(max(0, n - M + 1), min(n + 1, N)):
            result[n] += x[k] * h[n - k]
    
    return result

def dtft(signal, omega):
    """Дискретно-временное преобразование Фурье (пункт 2)"""
    N = len(signal)
    X = np.zeros(len(omega), dtype=complex)
    
    for k, w in enumerate(omega):
        for n in range(N):
            X[k] += signal[n] * np.exp(-1j * w * n)
    
    return X

def idtft(X, omega, n):
    """Обратное дискретно-временное преобразование Фурье (пункт 2)"""
    N = len(omega)
    x = np.zeros(len(n), dtype=complex)
    
    for m, time_point in enumerate(n):
        for k in range(N):
            x[m] += X[k] * np.exp(1j * omega[k] * time_point)
        x[m] /= (2 * np.pi)
    
    return x

def shift_signal(signal, shift):
    """Сдвиг сигнала (пункт 3)"""
    return np.roll(signal, shift)

def fft_manual(x):
    """Ручная реализация БПФ с прореживанием по времени (пункт 4)"""
    N = len(x)
    
    if N <= 1:
        return x
    
    # Разделение на четные и нечетные отсчеты
    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])
    
    # Поворачивающие множители
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    
    # Объединение результатов
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def compute_amplitude_phase_spectrum(dft):
    """Вычисление амплитудного и фазового спектра (пункт 7)"""
    amplitude = np.abs(dft)
    phase = np.angle(dft)
    return amplitude, phase

def compute_power_spectral_density(dft, N):
    """Вычисление спектральной плотности мощности (пункт 5)"""
    return np.abs(dft)**2 / N

def compare_reconstruction_methods(signal, methods_data):
    """Сравнение методов восстановления сигнала (пункт 8)"""
    errors = {}
    for method_name, reconstructed in methods_data.items():
        errors[method_name] = np.mean((signal - np.real(reconstructed))**2)
    return errors


def main():
    # Создаем папки для результатов
    os.makedirs('results_punct1', exist_ok=True)
    os.makedirs('results_punct2', exist_ok=True)
    os.makedirs('results_punct3', exist_ok=True)
    
    # Общие параметры сигнала   
    N = 1024 # Количество точек дискретизации
    x_min, x_max = -4.0, 4.0 # Интервал анализа
    x = np.linspace(x_min, x_max, N) 
    dx = (x_max - x_min)/N   # Шаг дискретизации = 0.0078
    
    print("=== ЕДИНАЯ ПРОГРАММА ДЛЯ ВСЕХ ПУНКТОВ ЗАДАНИЯ ===")
    
    # ПУНКТ 1: Преобразование Фурье
    print("\n" + "="*50)
    print("ПУНКТ 1: Преобразование Фурье исходной функции")
    print("Прямое и обратное преобразование, их свертка, оценка схожести")
    print("="*50)
    
    signal = f12(x)
    
    # 1.1 Исходный сигнал
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, 'b-', linewidth=2)
    plt.title('Исходный сигнал: f(x) = cos(3x⁴ - 3x³ - 3x² + 3x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/1_исходный_сигнал.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1.1 сохранен")
    
    # 1.2 Прямое преобразование Фурье
    dft = fft(signal)
    frequencies = fftfreq(N, d=dx)
    
    positive_freq = frequencies > 0
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[positive_freq], np.abs(dft[positive_freq]), 'r-', linewidth=2)
    plt.title('Амплитудный спектр Фурье')
    plt.xlabel('Частота ω')
    plt.ylabel('|X(ω)|')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/2_амплитудный_спектр.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1.2 сохранен")
    
    # 1.3 Обратное преобразование Фурье
    reconstructed = ifft(dft)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, 'b-', linewidth=2, label='Исходный')
    plt.plot(x, np.real(reconstructed), 'r--', linewidth=1.5, label='Восстановленный')
    plt.title('Сравнение исходного и восстановленного сигнала')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results_punct1/3_сравнение_сигналов.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1.3 сохранен")
    
    # 1.4 Свертка
    sigma = 0.5
    kernel = np.exp(-x**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    conv_time = convolution_time_domain(signal, kernel)
    
    plt.figure(figsize=(10, 6))
    conv_x = np.linspace(2*x_min, 2*x_max, len(conv_time))
    plt.plot(conv_x[:300], conv_time[:300], 'purple', linewidth=2)
    plt.title('Свертка сигнала с гауссовым ядром')
    plt.xlabel('x')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/4_свертка_сигнала.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1.4 сохранен")
    
    # 1.5 Оценка схожести
    mse = np.mean((signal - np.real(reconstructed))**2)
    error = signal - np.real(reconstructed)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, 'k-', linewidth=1.5)
    plt.title(f'Ошибка восстановления (MSE = {mse:.2e})')
    plt.xlabel('x')
    plt.ylabel('Ошибка Δf(x)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/5_ошибка_восстановления.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 1.5 сохранен")
    
    print(f"Результаты пункта 1: MSE = {mse:.2e}")
    
    # ПУНКТ 2: Дискретно-временное преобразование Фурье
    print("\n" + "="*50)
    print("ПУНКТ 2: Дискретно-временное преобразование Фурье")
    print("Прямое и обратное ДВПФ, их свертка")
    print("="*50)
    
    # 2.1 Дискретный сигнал
    plt.figure(figsize=(10, 6))
    plt.stem(x[::16], signal[::16], 'b-', basefmt=" ")
    plt.plot(x, signal, 'r-', alpha=0.3)
    plt.title('Дискретный исходный сигнал')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct2/1_дискретный_сигнал.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2.1 сохранен")
    
    # 2.2 Прямое ДВПФ
    omega = np.linspace(-np.pi, np.pi, 512)  # Частотная ось для ДВПФ
    X_dtft = dtft(signal, omega)
    
    plt.figure(figsize=(10, 6))
    plt.plot(omega, np.abs(X_dtft), 'g-', linewidth=2)
    plt.title('Дискретно-временное преобразование Фурье (ДВПФ)')
    plt.xlabel('Цифровая частота ω')
    plt.ylabel('|X(e^{jω})|')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct2/2_двпф_спектр.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2.2 сохранен")
    
    # 2.3 Обратное ДВПФ
    n_range = np.arange(N)
    reconstructed_dtft = idtft(X_dtft, omega, n_range)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, 'b-', linewidth=2, label='Исходный')
    plt.plot(x, np.real(reconstructed_dtft), 'g--', linewidth=1.5, label='Восстановленный (ДВПФ)')
    plt.title('Восстановление сигнала обратным ДВПФ')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results_punct2/3_восстановление_двпф.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2.3 сохранен")
    
    # 2.4 Свертка через ДВПФ
    H_dtft = dtft(kernel, omega)
    Y_dtft = X_dtft * H_dtft
    
    # Обратное ДВПФ для получения свертки
    conv_dtft = idtft(Y_dtft, omega, np.arange(len(conv_time)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(conv_x[:300], np.real(conv_dtft[:300]), 'orange', linewidth=2)
    plt.title('Свертка через ДВПФ')
    plt.xlabel('x')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct2/4_свертка_двпф.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 2.4 сохранен")
    
    # ПУНКТ 3: Сдвиги и свертка со сдвигом
    print("\n" + "="*50)
    print("ПУНКТ 3: Сдвиги. Свертка исходного сигнала и сдвига")
    print("="*50)
    
    # 3.1 Сдвиг сигнала
    shift = 100
    signal_shifted = shift_signal(signal, shift)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, 'b-', linewidth=2, label='Исходный')
    plt.plot(x, signal_shifted, 'r-', linewidth=2, label=f'Сдвинутый на {shift} точек')
    plt.title('Сдвиг сигнала')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results_punct3/1_сдвиг_сигнала.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 3.1 сохранен")
    
    # 3.2 Свертка со сдвинутым сигналом
    conv_shifted = convolution_time_domain(signal, signal_shifted)
    
    plt.figure(figsize=(10, 6))
    conv_shift_x = np.linspace(2*x_min, 2*x_max, len(conv_shifted))
    plt.plot(conv_shift_x[:300], conv_shifted[:300], 'purple', linewidth=2)
    plt.title('Свертка исходного сигнала со сдвинутым')
    plt.xlabel('x')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct3/2_свертка_со_сдвигом.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 3.2 сохранен")
    
    # 3.3 Сравнение спектров исходного и сдвинутого сигнала
    dft_shifted = fft(signal_shifted)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[positive_freq], np.abs(dft[positive_freq]), 'b-', linewidth=2, label='Исходный')
    plt.plot(frequencies[positive_freq], np.abs(dft_shifted[positive_freq]), 'r-', linewidth=1.5, label='Сдвинутый')
    plt.title('Сравнение амплитудных спектров')
    plt.xlabel('Частота ω')
    plt.ylabel('|X(ω)|')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('results_punct3/3_сравнение_спектров.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 3.3 сохранен")
    
    # ПУНКТ 4: Быстрое преобразование Фурье сдвигов
    print("\n" + "="*50)
    print("ПУНКТ 4: Быстрое преобразование Фурье сдвигов")
    print("="*50)
    
    # 4.1 БПФ исходного и сдвинутого сигнала
    # Используем scipy FFT для скорости (ручная реализация очень медленная для больших N)
    fft_original = fft(signal)
    fft_shifted = fft(signal_shifted)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(frequencies[positive_freq], np.abs(fft_original[positive_freq]), 'b-', linewidth=2, label='Исходный')
    plt.plot(frequencies[positive_freq], np.abs(fft_shifted[positive_freq]), 'r-', linewidth=1.5, label='Сдвинутый')
    plt.title('БПФ исходного и сдвинутого сигнала (амплитуда)')
    plt.xlabel('Частота ω')
    plt.ylabel('|X(ω)|')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[positive_freq], np.angle(fft_original[positive_freq]), 'b-', linewidth=2, label='Исходный')
    plt.plot(frequencies[positive_freq], np.angle(fft_shifted[positive_freq]), 'r-', linewidth=1.5, label='Сдвинутый')
    plt.title('БПФ исходного и сдвинутого сигнала (фаза)')
    plt.xlabel('Частота ω')
    plt.ylabel('∠X(ω)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_punct3/4_бпф_сдвигов.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 4.1 сохранен")
    
    # ПУНКТ 5: Спектральная плотность сигнала
    print("\n" + "="*50)
    print("ПУНКТ 5: Спектральная плотность сигнала")
    print("="*50)
    
    psd = compute_power_spectral_density(dft, N)
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies[positive_freq], psd[positive_freq], 'purple', linewidth=2)
    plt.title('Спектральная плотность мощности')
    plt.xlabel('Частота ω')
    plt.ylabel('S(ω)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/6_спектральная_плотность.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 5.1 сохранен")
    
    # ПУНКТ 7: Амплитудный и фазовый спектр
    print("\n" + "="*50)
    print("ПУНКТ 7: Амплитудный и фазовый спектр")
    print("="*50)
    
    amplitude, phase = compute_amplitude_phase_spectrum(dft)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(frequencies[positive_freq], amplitude[positive_freq], 'b-', linewidth=2)
    plt.title('Амплитудный спектр')
    plt.xlabel('Частота ω')
    plt.ylabel('|X(ω)|')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[positive_freq], phase[positive_freq], 'r-', linewidth=2)
    plt.title('Фазовый спектр')
    plt.xlabel('Частота ω')
    plt.ylabel('∠X(ω) [рад]')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_punct1/7_амплитудный_фазовый_спектр.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 7.1 сохранен")
    
    # ПУНКТ 8: Восстановление сигнала по спектру
    print("\n" + "="*50)
    print("ПУНКТ 8: Восстановление сигнала по спектру")
    print("="*50)
    
    # Восстановление с ограничением по частотам
    cutoff_freq = 0.3  # Ограничение верхней частоты
    dft_limited = dft.copy()
    dft_limited[np.abs(frequencies) > cutoff_freq] = 0
    reconstructed_limited = ifft(dft_limited)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, signal, 'b-', linewidth=2, label='Исходный')
    plt.plot(x, np.real(reconstructed), 'r--', linewidth=1.5, label='Полное восстановление')
    plt.plot(x, np.real(reconstructed_limited), 'g--', linewidth=1.5, label=f'Восстановление (f ≤ {cutoff_freq} Гц)')
    plt.title('Восстановление сигнала по ограниченному спектру')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    error_full = signal - np.real(reconstructed)
    error_limited = signal - np.real(reconstructed_limited)
    plt.plot(x, error_full, 'r-', linewidth=1.5, label='Ошибка полного восстановления')
    plt.plot(x, error_limited, 'g-', linewidth=1.5, label='Ошибка ограниченного восстановления')
    plt.title('Сравнение ошибок восстановления')
    plt.xlabel('x')
    plt.ylabel('Ошибка Δf(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_punct1/8_восстановление_по_спектру.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ График 8.1 сохранен")
    
    # Сравнение методов восстановления
    methods_data = {
        'DFT полный': reconstructed,
        'DFT ограниченный': reconstructed_limited,
        'ДВПФ': reconstructed_dtft
    }
    errors_comparison = compare_reconstruction_methods(signal, methods_data)
    
    print("\nСравнение методов восстановления (MSE):")
    for method, error in errors_comparison.items():
        print(f"  {method}: {error:.2e}")


if __name__ == "__main__":
    main()