import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

def build_histogram(image):
    """Построение гистограммы яркости изображения"""
    height, width = image.shape
    histogram = np.zeros(256, dtype=int)
    
    for i in range(height):
        for j in range(width):
            pixel_value = int(image[i, j])
            histogram[pixel_value] += 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(256), histogram, color='gray')
    plt.title('Гистограмма яркости')
    plt.xlabel('Уровень яркости')
    plt.ylabel('Количество пикселей')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    return histogram

def calculate_moments(image, histogram):
    """Вычисление начальных и центральных моментов изображения"""
    height, width = image.shape
    total_pixels = height * width
    
    m1 = np.sum([h * histogram[h] for h in range(256)]) / total_pixels
    m2 = np.sum([h**2 * histogram[h] for h in range(256)]) / total_pixels
    m3 = np.sum([h**3 * histogram[h] for h in range(256)]) / total_pixels
    m4 = np.sum([h**4 * histogram[h] for h in range(256)]) / total_pixels
    
    u2 = m2 - m1**2
    u3 = m3 - 3*m1*m2 + 2*m1**3
    u4 = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4
    
    sigma = np.sqrt(u2) if u2 > 0 else 0
    g1 = u3 / (sigma**3) if sigma != 0 else 0
    g2 = u4 / (sigma**4) - 3 if sigma != 0 else 0
    
    moments = {
        'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
        'u1': 0, 'u2': u2, 'u3': u3, 'u4': u4,
        'sigma': sigma,
        'g1': g1, 'g2': g2
    }
    
    print("Начальные моменты:")
    print(f"m1 (среднее значение): {m1:.2f}")
    print(f"m2: {m2:.2f}")
    print(f"m3: {m3:.2f}")
    print(f"m4: {m4:.2f}")
    
    print("\nЦентральные моменты:")
    print(f"u2 (дисперсия): {u2:.2f}")
    print(f"u3: {u3:.2f}")
    print(f"u4: {u4:.2f}")
    
    print("\nДополнительные характеристики:")
    print(f"Стандартное отклонение: {sigma:.2f}")
    print(f"Коэффициент асимметрии: {g1:.2f}")
    print(f"Коэффициент эксцесса: {g2:.2f}")
    
    return moments

def calculate_entropy_and_redundancy(histogram):
    """Вычисление энтропии и избыточности изображения"""
    total_pixels = np.sum(histogram)
    h_max = 255
    h_min = 0
    
    entropy = 0
    for h in range(256):
        if histogram[h] > 0:
            p = histogram[h] / total_pixels
            entropy -= p * math.log2(p)
    
    max_entropy = math.log2(h_max - h_min + 1)
    redundancy = 1 - (entropy / max_entropy) if max_entropy != 0 else 0
    
    print(f"Энтропия: {entropy:.4f} бит/пиксель")
    print(f"Максимальная энтропия: {max_entropy:.4f} бит/пиксель")
    print(f"Избыточность: {redundancy:.4f} ({redundancy*100:.2f}%)")
    
    return entropy, redundancy

def toroidal_padding(image, pad_size):
    """Добавление торOIDальной паддинга к изображению"""
    height, width = image.shape
    
    padded_image = np.zeros((height + 2*pad_size, width + 2*pad_size), dtype=image.dtype)
    padded_image[pad_size:pad_size+height, pad_size:pad_size+width] = image
    
    for i in range(height):
        padded_image[pad_size+i, :pad_size] = image[i, width-pad_size:]
        padded_image[pad_size+i, width+pad_size:] = image[i, :pad_size]
    
    for j in range(width):
        padded_image[:pad_size, pad_size+j] = image[height-pad_size:, j]
        padded_image[height+pad_size:, pad_size+j] = image[:pad_size, j]
    
    padded_image[:pad_size, :pad_size] = image[height-pad_size:, width-pad_size:]
    padded_image[:pad_size, width+pad_size:] = image[height-pad_size:, :pad_size]
    padded_image[height+pad_size:, :pad_size] = image[:pad_size, width-pad_size:]
    padded_image[height+pad_size:, width+pad_size:] = image[:pad_size, :pad_size]
    
    return padded_image

def apply_filter(image, kernel, padding_type='toroidal'):
    """Применение фильтра к изображению с учетом граничных условий"""
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_size = k_height // 2
    
    if padding_type == 'toroidal':
        padded_image = toroidal_padding(image, pad_size)
    else:
        padded_image = np.pad(image, pad_size, mode='constant')
    
    filtered_image = np.zeros_like(image, dtype=float)
    
    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+k_height, j:j+k_width]
            filtered_image[i, j] = np.sum(window * kernel)
    
    return filtered_image

def apply_affine_transform(image, A, t, output_shape=None):
    """
    Применяет аффинное преобразование с T-схемой (тороидальной) и билинейной интерполяцией.
    Обратное отображение + T-границы + интерполяция.
    """
    h, w = image.shape
    if output_shape is None:
        output_shape = (h, w)
    out_h, out_w = output_shape
    
    # Расширяем A и t до однородных координат
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = t
    M_inv = np.linalg.inv(M)
    
    result = np.zeros((out_h, out_w), dtype=np.float32)
    
    for i in range(out_h):
        for j in range(out_w):
            # Обратное отображение: из выходного -> исходное
            src = M_inv @ np.array([j, i, 1.0])
            x, y = src[0], src[1]
            
            # T-схема: wrap по модулю
            x = x % w
            y = y % h
            
            # Билинейная интерполяция
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = (x0 + 1) % w
            y1 = (y0 + 1) % h
            dx = x - x0
            dy = y - y0
            
            # Значения четырёх соседей
            I00 = image[y0, x0]
            I01 = image[y0, x1]
            I10 = image[y1, x0]
            I11 = image[y1, x1]
            
            # Билинейная интерполяция
            val = (
                I00 * (1 - dx) * (1 - dy) +
                I01 * dx * (1 - dy) +
                I10 * (1 - dx) * dy +
                I11 * dx * dy
            )
            result[i, j] = val
    
    return result

def apply_affine_transformations_demo(image):
    """
    Демонстрация трёх аффинных преобразований:
    1. Поворот на 30°
    2. Масштабирование 1.2x
    3. Горизонтальный сдвиг
    Все — с T-схемой.
    """
    h, w = image.shape
    cx, cy = w / 2, h / 2
    results = {}

    # 1. Поворот на 30°
    theta = np.radians(30)
    A_rot = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    # Компенсация смещения центра
    t_rot = np.array([cx, cy]) - A_rot @ np.array([cx, cy])
    rotated = apply_affine_transform(image, A_rot, t_rot)
    results['Поворот 30°'] = rotated

    # 2. Масштабирование 1.2x
    scale = 1.2
    A_scale = np.array([[scale, 0],
                        [0, scale]])
    t_scale = np.array([cx, cy]) - A_scale @ np.array([cx, cy])
    scaled = apply_affine_transform(image, A_scale, t_scale)
    results['Масштаб 1.2x'] = scaled

    # 3. Сдвиг (shear) по X
    k = 0.3
    A_shear = np.array([[1, k],
                        [0, 1]])
    t_shear = np.array([0, 0])  # без компенсации
    sheared = apply_affine_transform(image, A_shear, t_shear)
    results['Сдвиг (k=0.3)'] = sheared

    # Визуализация
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное')
    plt.axis('off')

    for idx, (name, img) in enumerate(results.items(), 2):
        plt.subplot(1, 4, idx)
        plt.imshow(img, cmap='gray')
        plt.title(name)
        plt.axis('off')
        
        # Вывод min/max
        print(f"{name}: min={np.min(img):.2f}, max={np.max(img):.2f}")
    
    plt.tight_layout()
    plt.show()
    
    return results

def median_filter(image, window_size=3):
    """Медианный фильтр с торOIDальной обработкой"""
    height, width = image.shape
    padded = toroidal_padding(image, window_size//2)
    result = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            window = padded[i:i+window_size, j:j+window_size]
            result[i, j] = np.median(window)
    
    return result

def sobel_filter(image):
    """Фильтр Собела с модулем"""
    height, width = image.shape
    padded = toroidal_padding(image, 1)
    result = np.zeros_like(image, dtype=float)
    
    for i in range(height):
        for j in range(width):
            X = (padded[i, j+2] + 2*padded[i+1, j+2] + padded[i+2, j+2]) - \
                (padded[i, j] + 2*padded[i+1, j] + padded[i+2, j])
            
            Y = (padded[i, j] + 2*padded[i, j+1] + padded[i, j+2]) - \
                (padded[i+2, j] + 2*padded[i+2, j+1] + padded[i+2, j+2])
            
            result[i, j] = abs(X) + abs(Y)
    
    return result

def apply_all_filters(image):
    """Применение всех указанных фильтров к изображению"""
    filters = {
        'Сглаживающий': (1/10) * np.array([[1, 1, 1],
                                           [1, 2, 1],
                                           [1, 1, 1]]),
        
        'Подчеркивание линий': (1/16) * np.array([[2, 1, 2],
                                                 [1, 4, 1],
                                                 [2, 1, 2]]),
        
        'ВЧ': np.array([[1, -2, 1],
                        [-2, 5, -2],
                        [1, -2, 1]]),
        
        'Градиент ЮВ': np.array([[-1, -1, 1],
                                 [-1, -2, 1],
                                 [1, 1, 1]]),
        
        'Лаплас': np.array([[1, -2, 1],
                            [-2, 4, -2],
                            [1, -2, 1]]),
    }
    
    results = {}
    
    for name, kernel in filters.items():
        filtered = apply_filter(image, kernel)
        results[name] = filtered
    
    results['Медианный'] = median_filter(image)
    results['Собел'] = sobel_filter(image)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    for idx, (name, filtered) in enumerate(results.items(), 2):
        plt.subplot(3, 3, idx)
        if name in ['ВЧ', 'Лаплас', 'Градиент ЮВ', 'Собел']:
            # Для фильтров с отрицательными значениями — показываем модуль
            plt.imshow(np.abs(filtered), cmap='gray')
        else:
            # Для остальных — обрезаем в [0, 255]
            plt.imshow(np.clip(filtered, 0, 255), cmap='gray')
    
        print(f"{name}: min={np.min(filtered):.2f}, max={np.max(filtered):.2f}")
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return results

def btc_encoding(image, block_size=4):
    """Усеченное блочное кодирование изображения"""
    height, width = image.shape
    encoded_image = np.zeros_like(image, dtype=float)
    blocks_info = []
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            block_height, block_width = block.shape
            
            if block_height != block_size or block_width != block_size:
                continue
            
            C = np.mean(block)
            E = np.mean(block**2)
            sigma2 = E - C**2
            sigma = np.sqrt(sigma2) if sigma2 > 0 else 0
            
            d = C
            q = np.sum(block >= d)
            p = block_size * block_size
            
            if q == 0 or q == p:
                a = C
                b = C
            else:
                a = C - sigma * np.sqrt(q / (p - q))
                b = C + sigma * np.sqrt((p - q) / q)
            
            quantized_block = np.where(block < d, a, b)
            encoded_image[i:i+block_size, j:j+block_size] = quantized_block
            
            blocks_info.append({
                'position': (i, j),
                'C': C,
                'sigma': sigma,
                'd': d,
                'q': q,
                'a': a,
                'b': b
            })
    
    mse = np.mean((image - encoded_image)**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Пиковое отношение сигнал/шум (PSNR): {psnr:.2f} дБ")
    
    return encoded_image, blocks_info, mse, psnr


def get_image_path():
    """Запрос относительного пути к изображению с проверкой"""
    while True:
        image_path = input("Введите относительный путь к изображению (например, 'images/wolf.png'): ").strip()
        if not image_path:
            print("Путь не может быть пустым.")
            continue
        try:
            # Проверяем существование файла
            if not os.path.exists(image_path):
                print(f"Файл '{image_path}' не найден. Попробуйте снова.")
                continue
            return image_path
        except Exception as e:
            print(f"Ошибка при проверке файла: {e}. Попробуйте снова.")

# Основной код выполнения задания
def main():
    image_path = get_image_path()
    image = Image.open(image_path).convert('L')
    image_array = np.array(image, dtype=np.float32)

    print("1. Построение гистограммы яркости")
    histogram = build_histogram(image_array)

    print("\n2. Вычисление начальных и центральных моментов")
    moments = calculate_moments(image_array, histogram)

    print("\n3. Вычисление энтропии и избыточности")
    entropy, redundancy = calculate_entropy_and_redundancy(histogram)

    print("\n4. Усеченное блочное кодирование")
    encoded_image, blocks_info, mse, psnr = btc_encoding(image_array)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(encoded_image, cmap='gray')
    plt.title(f'Закодированное\nMSE: {mse:.2f}, PSNR: {psnr:.2f} дБ')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # === НАСТОЯЩИЙ ПУНКТ 5 ПО ЗАДАНИЮ ===
    print("\n5. Геометрические преобразования (аффинные)")
    affine_results = apply_affine_transformations_demo(image_array)

    # Опционально: фильтрация как пункт 6
    print("\n6. Локальная фильтрация")
    filtered_results = apply_all_filters(image_array)

if __name__ == "__main__":
    main()