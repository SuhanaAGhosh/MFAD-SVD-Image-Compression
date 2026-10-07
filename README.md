# Image Compression using Singular Value Decomposition (SVD)

A Maths for AI project that implements **SVD from scratch** in NumPy and uses it to compress grayscale and colour images with low-rank approximations. We check our implementation against NumPy, measure reconstruction quality (Frobenius error, PSNR), and find out when SVD compression actually saves storage and when it doesn't.

---

## Contents

| Part | What it does |
|------|--------------|
| **P1** | SVD from scratch using the eigendecomposition of AᵀA, checked against `np.linalg.svd` |
| **P2** | Rank-k compression of a grayscale image, with storage savings |
| **P3** | Colour (RGB) compression: each channel compressed separately |
| **P4** | Error analysis: Frobenius error, PSNR, and the Eckart–Young identity |
| **P5** | Break-even analysis and failure cases |

---

## Theory in short

Any matrix **A** (m × n) can be written as

```
A = U Σ Vᵀ
```

where U and V have orthonormal columns and Σ is diagonal with singular values σ₁ ≥ σ₂ ≥ … ≥ 0.

Keeping only the top k singular values gives the **rank-k approximation**

```
A_k = U[:, :k] · diag(σ₁…σ_k) · Vᵀ[:k, :]
```

By the **Eckart–Young theorem**, this is the best rank-k approximation, and its error is exactly the energy in the discarded singular values:

```
‖A − A_k‖_F = √(σ²_{k+1} + σ²_{k+2} + …)
```

**Storage:** the original needs `m·n` values; the rank-k form needs `k·(m + n + 1)` values (k columns of U, k rows of Vᵀ, k singular values).

---

## P1 — SVD from scratch

```python
def my_svd(A, tol=1e-10):
    eigvals, V = np.linalg.eigh(A.T @ A)   # AᵀA is symmetric → real eigenvalues
    idx = np.argsort(eigvals)[::-1]         # sort largest first
    eigvals, V = eigvals[idx], V[:, idx]
    eigvals = np.clip(eigvals, 0, None)     # remove tiny negatives from rounding
    S = np.sqrt(eigvals)                    # σ = √λ
    keep = S > tol * S[0]                   # drop zero singular values
    S, V = S[keep], V[:, keep]
    U = (A @ V) / S                         # u_i = A v_i / σ_i
    return U, S, V.T
```

**Verification results**

| Test | Result |
|------|--------|
| Random 50×30 matrix: singular values match NumPy | ✅ True |
| Reconstruction error ‖B − UΣVᵀ‖ | 2.03 × 10⁻¹⁴ |
| 512×512 image: top 50 singular values match NumPy | ✅ True |
| Max difference across all singular values | 7.82 × 10⁻⁸ |
| Rank-50 Frobenius error (ours vs NumPy) | 5080.385268980226 vs 5080.385268980224 |

The per-value difference grows slightly for the smallest singular values. That is expected: forming AᵀA squares the condition number, so tiny σ's lose precision. The large singular values, which carry the image, are accurate.

---

## P2 — Grayscale compression (512 × 512)

Original storage: **262,144** values

| k | Values stored | Compression ratio | Space saved |
|---|---------------|-------------------|-------------|
| 5 | 5,125 | 51.15× | 98.04% |
| 20 | 20,500 | 12.79× | 92.18% |
| 50 | 51,250 | 5.12× | 80.45% |
| 100 | 102,500 | 2.56× | 60.90% |

- **k = 5:** only the broad layout (sky vs. ground), heavy blur and streaks
- **k = 20:** clouds and horizon become recognisable
- **k = 50:** close to the original; fine grass texture is softer
- **k = 100:** visually almost the same as the original

The singular values fall quickly (from ~10⁵ to ~10³ within the first ~20), which is why a small k already captures most of the image.

---

## P3 — Colour (RGB) compression

The R, G and B channels are separate 512×512 matrices. Each one is decomposed once with `my_svd` and then reused for every k. Reconstructed values are clipped to 0–255 and cast to `uint8`. The `compress_rgb` function also allows a different k per channel.

| k | Values stored | Original | Fraction of original |
|---|---------------|----------|----------------------|
| 5 | 15,375 | 786,432 | 2.0% |
| 20 | 61,500 | 786,432 | 7.8% |
| 50 | 153,750 | 786,432 | 19.6% |
| 100 | 307,500 | 786,432 | 39.1% |

---

## P4 — Error analysis

We compare the measured Frobenius error with the Eckart–Young prediction (the square root of the sum of the discarded σ²):

| k | ‖A − A_k‖_F | √(Σ tail σ²) | Abs. difference | PSNR (dB) |
|---|-------------|--------------|-----------------|-----------|
| 5 | 10788.1072 | 10788.1072 | 4.00e-11 | 21.66 |
| 10 | 9395.5421 | 9395.5421 | 4.18e-11 | 22.86 |
| 20 | 7704.7768 | 7704.7768 | 5.46e-11 | 24.58 |
| 40 | 5746.1689 | 5746.1689 | 7.09e-11 | 27.13 |
| 80 | 3661.3073 | 3661.3073 | 1.06e-10 | 31.05 |

The two columns agree to about 10⁻¹⁰, which confirms the theorem numerically.

PSNR is computed on the clipped (displayable) image:

```
PSNR = 10 · log₁₀(255² / MSE)
```

Across k = 1…256, the Frobenius error drops steeply at first and then levels off, while PSNR rises from about 18 dB to about 45 dB. Most of the quality gain comes from the first few dozen singular values.

---

## P5 — When is SVD compression worth it?

Compression only helps when

```
k · (m + n + 1) < m · n
```

For our 512×512 image:

| Quantity | Value |
|----------|-------|
| Break-even k | 255.75 |
| Largest beneficial k | 255 |
| At k = 50 | 5.12× compression, 80.45% saved: **beneficial** |

Above k ≈ 256, the "compressed" form is larger than the original image.

### Failure case 1: small image with a large k

A random 20×20 matrix compressed with k = 15:

| Original | SVD storage | Relative error | Saving |
|----------|-------------|----------------|--------|
| 400 | 615 | 0.0596 | **−53.75%** |

The reconstruction is good, but it takes **more** storage than the original.

### Failure case 2: text images

Images of text have sharp edges, so their singular values decay slowly. A small k leaves the text blurry or unreadable, and a large k removes most of the storage benefit. Low-rank SVD works best on smooth, natural images.

---

## Running the notebook

1. Open the notebook in Google Colab (or Jupyter).
2. Upload an image named **`img.jpg`** (the notebook uses `files.upload()` in Colab).
3. Run all cells in order. `U, S, Vt, m, n` are computed once and shared by P2, P4 and P5, so don't reassign them.

### Requirements

```
numpy        # tested on 2.1.3
matplotlib
pillow
```

```bash
pip install numpy matplotlib pillow
```

---

## Key takeaways

- An SVD built on the eigendecomposition of AᵀA matches NumPy to about 10⁻⁸ and gives identical reconstructions.
- Natural images are approximately low-rank. Keeping 50 of 512 singular values saves about 80% of storage and still looks close to the original.
- The Eckart–Young theorem holds exactly in practice: the reconstruction error equals the energy in the discarded singular values.
- SVD compression is not always useful. It loses on small matrices, at large k, and on edge-heavy content such as text.

---
