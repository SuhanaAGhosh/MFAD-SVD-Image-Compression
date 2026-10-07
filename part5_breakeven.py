import numpy as np

print("===============================")
print("SVD BREAK-EVEN DEMO")
print("===============================")

# User input
m = int(input("Enter image height (m): "))
n = int(input("Enter image width (n): "))
k = int(input("Enter rank k: "))

# Simple validation
if m <= 0 or n <= 0 or k <= 0:
    print("Error: m, n and k must be positive.")
    exit()
if k > min(m, n):
    print("Error: k cannot be more than min(m, n) =", min(m, n))
    exit()

# Storage formulas
original = m * n                  # original storage = m*n
svd = k * (m + n + 1)             # rank-k SVD storage = k(m+n+1)
threshold = original / (m + n + 1)        # break-even: k(m+n+1) = mn
largest_k = (original - 1) // (m + n + 1) # largest integer k with k(m+n+1) < mn (strict)
ratio = original / svd
saving = (1 - svd / original) * 100

print("\nFormula: beneficial only if k*(m+n+1) < m*n")
print("Original storage:", original)
print("SVD storage:", svd)
print(f"Break-even k: {threshold:.2f}")
print("Largest beneficial k:", largest_k)
print(f"Compression ratio: {ratio:.2f}")
print(f"Storage saving: {saving:.2f}%")
if svd < original:
    print("Status: BENEFICIAL")
else:
    print("Status: NOT BENEFICIAL")

# Failure case: SVD works fine, but k is too large for a small image
m2, n2, k2 = 20, 20, 15
np.random.seed(0)
img = np.random.rand(m2, n2)
U, S, Vt = np.linalg.svd(img)
approx = U[:, :k2] @ np.diag(S[:k2]) @ Vt[:k2, :]
error = np.linalg.norm(img - approx) / np.linalg.norm(img)

original2 = m2 * n2
svd2 = k2 * (m2 + n2 + 1)
saving2 = (1 - svd2 / original2) * 100

print("\nFailure case")
print(f"Image size: {m2} x {n2}")
print("Chosen k:", k2)
print("Original storage:", original2)
print("SVD storage:", svd2)
print(f"Relative reconstruction error: {error:.4f}")
print(f"Storage saving: {saving2:.2f}%")
if svd2 < original2:
    print("Status: BENEFICIAL")
else:
    print("Status: NOT BENEFICIAL")
print("Reconstruction is good, but SVD storage is larger than the original.")
