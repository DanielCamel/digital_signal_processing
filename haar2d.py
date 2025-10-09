import numpy as np
import matplotlib.pyplot as plt

def rowTransform(row, magnitude):
    result = []
    length = len(row)
    start = 0
    step = length
    for i in range(magnitude + 1):
        for j in range(2**(i)):
            start = 2 * step * j
            if start >= length:
                break
            count = 0
            total = 0
            for k in range(step):
                total += row[start+k]
                count += 1
                if count >= step:
                    break
            if step != length:
                start += step
                count = 0
                for k in range(step):
                    total += (-1) * row[start+k]
                    count += 1
                    if count >= step:
                        break
                total = total * (2 ** (0.5*(1 - i))) / (2 * step)
            else:
                total = total / step
            result.append(total)
        step = step // 2
    return result

def pretty_print(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] < 0:
                print("-", end='')
            else:
                print(' ', end='')
            if j != len(matrix[0]) - 1:
                print(f' {abs(round(matrix[i][j], 5)):<7} ' + " ", end='')
            else:
                print(f' {abs(round(matrix[i][j], 5)):<7} ' + " ")

picture = []
picture.append([1, 1, 0, 1])
picture.append([0, 0, 0, 0])
picture.append([1, 0, 0, 1])
picture.append([1, 0, 0, 1])

print('Исходный паттерн: ')
pretty_print(picture)

fig, ax = plt.subplots()
ax.imshow(picture, cmap='gray', vmin=0, vmax=1)
plt.show()

print()

for i in range(len(picture)):
    picture[i] = rowTransform(picture[i], 2)
print('Паттерн после разложения рядов: ')
pretty_print(picture)

fig, ax = plt.subplots()
ax.imshow(picture, cmap='gray', vmin=0, vmax=1)
plt.show()

print()

for i in range(len(picture[0])):
    media = []
    for j in range(len(picture)):
        media.append(picture[j][i])
    mediaNew = rowTransform(media, 2)
    for j in range(len(picture)):
        picture[j][i] = mediaNew[j]

print('Паттерн после последующего разложения столбцов: ')
pretty_print(picture)

fig, ax = plt.subplots()
ax.imshow(picture, cmap='gray', vmin=0, vmax=1)
plt.show()