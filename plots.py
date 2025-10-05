import numpy as np
import matplotlib.pyplot as plt


def phi(x):

    if isinstance(x, np.ndarray):
        y = np.zeros_like(x, dtype=float)
        mask = (x >= 0.0) & (x <= 1.0)
        y[mask] = 1.0
        return y
    if x < 0.0 or x > 1.0:
        return 0.0
    return 1.0


if __name__ == "__main__":

    dx = 0.1
    xs = np.arange(-0.2, 1.2 + 0.5 * dx, dx)
    ys = phi(xs)
    plt.figure(figsize=(7, 3))
    plt.step(xs, ys, where="post")
    plt.axhline(0, color="#888", linewidth=0.8)
    plt.axvline(0, color="#888", linewidth=0.8)
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.xticks(np.arange(-0.2, 1.21, dx))
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.title("Масштабирующая функция phi(x)")
    plt.xlabel("x")
    plt.ylabel("phi(x)")
    plt.tight_layout()
    plt.show()

