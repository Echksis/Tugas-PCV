import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# MEMBACA GAMBAR
# ==========================================

folder = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(folder, "earth.jpg")

img = cv2.imread(img_path)

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

print("Gambar berhasil dibaca!")


# ==========================================
# KONVERSI BGR KE RGB
# ==========================================

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================
# MEMBUAT SALT & PEPPER NOISE
# ==========================================

noisy_img = img.copy()

# Jumlah noise
jumlah_noise = int(0.08 * img.shape[0] * img.shape[1])

# Salt Noise (putih)
for i in range(jumlah_noise):
    y = np.random.randint(0, img.shape[0])
    x = np.random.randint(0, img.shape[1])
    noisy_img[y, x] = [255, 255, 255]

# Pepper Noise (hitam)
for i in range(jumlah_noise):
    y = np.random.randint(0, img.shape[0])
    x = np.random.randint(0, img.shape[1])
    noisy_img[y, x] = [0, 0, 0]

noisy_rgb = cv2.cvtColor(
    noisy_img,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 1. MEAN FILTER
# ==========================================

mean_filter = cv2.blur(
    noisy_img,
    (11, 11)
)

mean_rgb = cv2.cvtColor(
    mean_filter,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 2. MEDIAN FILTER
# ==========================================

median_filter = cv2.medianBlur(
    noisy_img,
    11
)

median_rgb = cv2.cvtColor(
    median_filter,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 3. GAUSSIAN FILTER
# ==========================================

gaussian_filter = cv2.GaussianBlur(
    noisy_img,
    (11, 11),
    0
)

gaussian_rgb = cv2.cvtColor(
    gaussian_filter,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# 4. SHARPENING
# ==========================================

kernel_sharpen = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1]
])

sharpen_filter = cv2.filter2D(
    img,
    -1,
    kernel_sharpen
)

sharpen_rgb = cv2.cvtColor(
    sharpen_filter,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# MENAMPILKAN HASIL
# ==========================================

plt.figure(figsize=(16, 10))


# 1. Gambar Asli
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Gambar Asli", fontsize=14)
plt.axis("off")


# 2. Noise
plt.subplot(2, 3, 2)
plt.imshow(noisy_rgb)
plt.title("Salt & Pepper Noise", fontsize=14)
plt.axis("off")


# 3. Mean
plt.subplot(2, 3, 3)
plt.imshow(mean_rgb)
plt.title("Mean Filter", fontsize=14)
plt.axis("off")


# 4. Median
plt.subplot(2, 3, 4)
plt.imshow(median_rgb)
plt.title("Median Filter", fontsize=14)
plt.axis("off")


# 5. Gaussian
plt.subplot(2, 3, 5)
plt.imshow(gaussian_rgb)
plt.title("Gaussian Filter", fontsize=14)
plt.axis("off")


# 6. Sharpening
plt.subplot(2, 3, 6)
plt.imshow(sharpen_rgb)
plt.title("Sharpening Filter", fontsize=14)
plt.axis("off")


plt.tight_layout()
plt.show()