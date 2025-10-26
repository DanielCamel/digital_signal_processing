import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import os

def f12(x):
    """функция f(x) = cos(3x^4 - 3x^3 - 3x^2 + 3x)"""
    return np.cos(3*x**4 - 3*x**3 - 3*x**2 + 3*x)

def convolution_time_domain(x, h):
    """свертка во временной области"""
    N = len(x)
    M = len(h)
    result = np.zeros(N + M - 1)
    
    for n in range(len(result)):
        for k in range(max(0, n - M + 1), min(n + 1, N)):
            result[n] += x[k] * h[n - k]
    
    return result

def dtft(signal, omega):
    """дискретно-временное преобразование Фурье"""
    N = len(signal)
    X = np.zeros(len(omega), dtype=complex)
    
    for k, w in enumerate(omega):
        for n in range(N):
            X[k] += signal[n] * np.exp(-1j * w * n)
    
    return X

def idtft(X, omega, n):
    """обратное дискретно-временное преобразование Фурье"""
    N = len(omega)
    x = np.zeros(len(n), dtype=complex)
    
    for m, time_point in enumerate(n):
        for k in range(N):
            x[m] += X[k] * np.exp(1j * omega[k] * time_point)
        x[m] /= (2 * np.pi)
    
    return x

def shift_signal(signal, shift):
    """сдвиг сигнала """
    return np.roll(signal, shift)

def fft_manual(x):
    """быстрое преобразование Фурье с прореживанием по времени"""
    N = len(x)
    
    if N <= 1:
        return x
    
    # разделение на чет/нечет отсчеты
    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])
    
    # поворачивающие множители
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    
    # объединение результатов
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

def compute_amplitude_phase_spectrum(dft):
    """вычисление амплитудного и фазового спектра """
    amplitude = np.abs(dft)
    phase = np.angle(dft)
    return amplitude, phase

def compute_power_spectral_density(dft, N):
    """вычисление спектральной плотности мощности"""
    return np.abs(dft)**2 / N

def compare_reconstruction_methods(signal, methods_data):
    """сравнение методов восстановления сигнала"""
    errors = {}
    for method_name, reconstructed in methods_data.items():
        errors[method_name] = np.mean((signal - np.real(reconstructed))**2)
    return errors

def reconstruct_with_harmonics(signal, num_harmonics, T):
    """
    Восстановление сигнала суммой гармоник
    """
    N = len(signal)
    t = np.linspace(0, T, N, endpoint=False)
    
    # вычисление коэффициентов Фурье
    C0 = np.mean(signal)  # Постоянная составляющая
    
    ak = np.zeros(num_harmonics)
    bk = np.zeros(num_harmonics)
    
    for k in range(1, num_harmonics + 1):
        ak[k-1] = 2/T * np.sum(signal * np.cos(2*np.pi*k*t/T)) * (T/N)
        bk[k-1] = 2/T * np.sum(signal * np.sin(2*np.pi*k*t/T)) * (T/N)
    
    # восстановление сигнала
    reconstructed = np.full_like(t, C0)
    
    for k in range(num_harmonics):
        reconstructed += ak[k] * np.cos(2*np.pi*(k+1)*t/T)
        reconstructed += bk[k] * np.sin(2*np.pi*(k+1)*t/T)
    
    return reconstructed, C0, ak, bk


# =========== Основная функция для выполнения анализа
def main():
    # папки для результатов
    os.makedirs('results_punct1', exist_ok=True)
    os.makedirs('results_punct2', exist_ok=True)
    os.makedirs('results_punct3', exist_ok=True)
    
    # общие параметры   
    N = 1024 # Количество точек дискретизации
    x_min, x_max = -4.0, 4.0 # Интервал анализа
    x = np.linspace(x_min, x_max, N) 
    dx = (x_max - x_min)/N   # Шаг дискретизации = 0.0078
    T = x_max - x_min     # Период функции для гармоник


    # преобразование Фурье
    print("Преобразование Фурье исходной функции")
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
    
    # 1.5 Оценка схожести
    mse = np.mean((signal - np.real(reconstructed))**2)
    error = signal - np.real(reconstructed)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, error, 'k-', linewidth=1.5)
    plt.title(f'Ошибка восстановления (MSE = {mse:.2e})')
    plt.xlabel('x')
    plt.ylabel('Ошибка delta(f(x))')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct1/5_ошибка_восстановления.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"MSE = {mse:.2e}")
    
    # дискретно-временное преобразование Фурье
    print("\n" + "="*50)
    print("Дискретно-временное преобразование Фурье")
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
    
    # 2.4 Свертка через ДВПФ
    H_dtft = dtft(kernel, omega)
    Y_dtft = X_dtft * H_dtft
    
    # обратное ДВПФ для получения свертки
    conv_dtft = idtft(Y_dtft, omega, np.arange(len(conv_time)))
    
    plt.figure(figsize=(10, 6))
    plt.plot(conv_x[:300], np.real(conv_dtft[:300]), 'orange', linewidth=2)
    plt.title('Свертка через ДВПФ')
    plt.xlabel('x')
    plt.ylabel('Амплитуда')
    plt.grid(True, alpha=0.3)
    plt.savefig('results_punct2/4_свертка_двпф.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # сдвиги и свертка со сдвигом
    print("\n" + "="*50)
    print("Сдвиги. Свертка исходного сигнала и сдвига")
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
    
    # быстрое преобразование Фурье сдвигов
    print("\n" + "="*50)
    print("Быстрое преобразование Фурье сдвигов")
    print("="*50)
    
    # БПФ исходного и сдвинутого сигнала
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
    plt.ylabel('arg(X(ω))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_punct3/4_бпф_сдвигов.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # спектральная плотность сигнала
    print("\n" + "="*50)
    print("Спектральная плотность сигнала")
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
    
    # амплитудный и фазовый спектр
    print("\n" + "="*50)
    print("Амплитудный и фазовый спектр")
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
    plt.ylabel('arg(X(ω)) [рад]')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_punct1/7_амплитудный_фазовый_спектр.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # восстановление сигнала по спектру
    print("\n" + "="*50)
    print("Восстановление сигнала по спектру")
    print("="*50)
    
    # восстановление с ограничением по частотам
    cutoff_freq = 0.3  # Ограничение верхней частоты
    dft_limited = dft.copy()
    dft_limited[np.abs(frequencies) > cutoff_freq] = 0
    reconstructed_limited = ifft(dft_limited)
    
    # восстановление суммой гармоник
    num_harmonics_list = [1, 3, 10, 50]  # Разное количество гармоник (1, 3, 10, 50)
    harmonic_reconstructions = {}
    
    for num_harmonics in num_harmonics_list:
        reconstructed_harmonic, C0, ak, bk = reconstruct_with_harmonics(signal, num_harmonics, T)
        harmonic_reconstructions[num_harmonics] = reconstructed_harmonic
        print(f"Восстановление с {num_harmonics} гармониками: C0={C0:.4f}")
    
    # График восстановления разным количеством гармоник
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, signal, 'k-', linewidth=3, label='Исходный сигнал', alpha=0.8)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (num_harmonics, reconstructed_harmonic) in enumerate(harmonic_reconstructions.items()):
        plt.plot(x, reconstructed_harmonic, '--', linewidth=1.5, 
                color=colors[i], label=f'{num_harmonics} гармоник')
    
    plt.title('восстановление сигнала суммой гармоник')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i, (num_harmonics, reconstructed_harmonic) in enumerate(harmonic_reconstructions.items()):
        error = signal - reconstructed_harmonic
        plt.plot(x, error, '-', linewidth=1.5, 
                color=colors[i], label=f'Ошибка ({num_harmonics} гарм.)')
    
    plt.title('Ошибки восстановления разным количеством гармоник')
    plt.xlabel('x')
    plt.ylabel('Ошибка delta(f(x))')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_punct1/8_восстановление_гармониками.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # чравнение всех методов восстановления
    plt.figure(figsize=(12, 8))
    
    methods_data = {
        'ДПФ': reconstructed,
        'ДПФ (ограниченное)': reconstructed_limited,
        '50 гармоник': harmonic_reconstructions[50]
    }
    
    plt.plot(x, signal, 'k-', linewidth=3, label='Исходный', alpha=0.8)
    
    for method_name, reconstructed_signal in methods_data.items():
        plt.plot(x, np.real(reconstructed_signal), '--', linewidth=1.5, label=method_name)
    
    plt.title('Сравнение методов восстановления сигнала')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results_punct1/9_сравнение_методов.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Сравнение ошибок методов
    methods_data_extended = {
        'ДПФ': reconstructed,
        'ДПФ (ограниченное)': reconstructed_limited,
        '1 гармоника': harmonic_reconstructions[1],
        '3 гармоники': harmonic_reconstructions[3],
        '10 гармоник': harmonic_reconstructions[10],
        '50 гармоник': harmonic_reconstructions[50]
    }
    
    errors_comparison = compare_reconstruction_methods(signal, methods_data_extended)
    
    print("\nСравнение методов восстановления (MSE):")
    for method, error in errors_comparison.items():
        print(f"  {method}: {error:.2e}")


if __name__ == "__main__":
    main()