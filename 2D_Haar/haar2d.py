# Haar 2D Transform Implementation
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def pair_transform_pairwise(vec):
    assert len(vec) % 2 == 0
    n = len(vec)//2
    s = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        x1 = vec[2*i]
        x2 = vec[2*i+1]
        s[i] = (x1 + x2) / sqrt2
        d[i] = (x1 - x2) / sqrt2
    return np.concatenate([s, d])

def inverse_pair_transform_pairwise(sd):
    n2 = len(sd)
    assert n2 % 2 == 0
    n = n2 // 2
    s = sd[:n]
    d = sd[n:]
    out = np.zeros(n2)
    for i in range(n):
        out[2*i]   = (s[i] + d[i]) / sqrt2
        out[2*i+1] = (s[i] - d[i]) / sqrt2
    return out

# symbolic helper
def to_symbolic(val, tol=1e-9):
    candidates = {
        "0": 0.0,
        "1": 1.0,
        "-1": -1.0,
        "1/2": 0.5,
        "-1/2": -0.5,
        "3/2": 1.5,
        "-3/2": -1.5,
        "sqrt(2)": sqrt2,
        "-sqrt(2)": -sqrt2,
        "1/sqrt(2)": 1.0/sqrt2,
        "-1/sqrt(2)": -1.0/sqrt2
    }
    for k,v in candidates.items():
        if abs(val - v) < tol:
            return k
    m = val * sqrt2
    if abs(m - round(m)) < tol:
        m_int = int(round(m))
        return f"{m_int}/√2" if m_int!=1 else "1/√2"
    return f"{val:.4f}"


A = np.array([
    [0,1,1,0],
    [1,0,1,1],
    [1,0,0,1],
    [0,1,1,1]
], dtype=float)
# transforms

sqrt2 = math.sqrt(2)
row_transformed = np.zeros_like(A)
for i in range(A.shape[0]):
    row_transformed[i, :] = pair_transform_pairwise(A[i, :])

W = np.zeros_like(A)
for j in range(A.shape[1]):
    col = row_transformed[:, j]
    W[:, j] = pair_transform_pairwise(col)

# inverse
intermediate = np.zeros_like(W)
for j in range(W.shape[1]):
    intermediate[:, j] = inverse_pair_transform_pairwise(W[:, j])

reconstructed = np.zeros_like(intermediate)
for i in range(intermediate.shape[0]):
    reconstructed[i, :] = inverse_pair_transform_pairwise(intermediate[i, :])

# prepare data_frames and print
df_A = pd.DataFrame(A)
df_row = pd.DataFrame(row_transformed)
df_W = pd.DataFrame(W)
df_intermediate = pd.DataFrame(intermediate)
df_recon = pd.DataFrame(reconstructed)

df_A_sym = df_A.applymap(lambda x: to_symbolic(x))
df_row_sym = df_row.applymap(lambda x: to_symbolic(x))
df_W_sym = df_W.applymap(lambda x: to_symbolic(x))
df_intermediate_sym = df_intermediate.applymap(lambda x: to_symbolic(x))
df_recon_sym = df_recon.applymap(lambda x: to_symbolic(x))

print("Original A:\n", df_A.to_string(index=False), "\n")
print("Row-wise transformed:\n", df_row.to_string(index=False), "\n")
print("Full 2D-transform W:\n", df_W.to_string(index=False), "\n")
print("Intermediate after inverse columns:\n", df_intermediate.to_string(index=False), "\n")
print("Reconstructed:\n", df_recon.to_string(index=False), "\n")

print("Original A (symbolic-like):\n", df_A_sym.to_string(index=False), "\n")
print("Row-wise transformed (symbolic-like):\n", df_row_sym.to_string(index=False), "\n")
print("Full 2D transform W (symbolic-like):\n", df_W_sym.to_string(index=False), "\n")

# plots
plt.figure(figsize=(4,4))
plt.title("Original pattern A")
plt.imshow(A, aspect='equal', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.title("After row-wise transform (numeric matrix)")
plt.imshow(row_transformed, aspect='equal', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.title("Full 2D transform W (numeric matrix)")
plt.imshow(W, aspect='equal', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(4,4))
plt.title("Reconstructed image (should equal original)")
plt.imshow(reconstructed, aspect='equal', interpolation='nearest')
plt.colorbar()
plt.tight_layout()
plt.show()

# verification (reconstruction error calculation)
diff = np.abs(reconstructed - A)
max_err = diff.max()
print(f"Maximum absolute reconstruction error: {max_err:e}")
if max_err < 1e-9:
    print("Reconstruction exact within numerical precision.")
else:
    print("Reconstruction differs; check computations.")

