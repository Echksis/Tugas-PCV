import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder = os.path.dirname(os.path.abspath(__file__))
path_gambar = os.path.join(folder, "earth.jpg")

# Membaca gambar asli berwarna
img_color = cv2.imread(path_gambar)

if img_color is None:
    print("Gambar tidak ditemukan!")
    exit()

# Mengubah gambar berwarna menjadi grayscale
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

height, width = img.shape

print("Gambar berhasil dibaca")
print("Ukuran gambar:", width, "x", height)

def histogram_manual(image):

    histogram = [0] * 256

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            pixel = image[i, j]

            histogram[pixel] += 1

    return histogram

def negative(image):

    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            hasil[i, j] = 255 - image[i, j]

    return hasil

def log_transform(image):

    hasil = np.zeros_like(image, dtype=np.uint8)

    c = 255 / np.log(256)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            r = image[i, j]

            s = c * np.log(1 + r)

            if s > 255:
                s = 255

            if s < 0:
                s = 0

            hasil[i, j] = int(s)

    return hasil

def gamma_transform(image, gamma):

    hasil = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            r = image[i, j] / 255.0

            s = 255 * (r ** gamma)

            if s > 255:
                s = 255

            if s < 0:
                s = 0

            hasil[i, j] = int(s)

    return hasil

def contrast_stretching(image):

    hasil = np.zeros_like(image, dtype=np.uint8)

    r_min = 255
    r_max = 0

    # Mencari nilai minimum dan maksimum
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            pixel = image[i, j]

            if pixel < r_min:
                r_min = pixel

            if pixel > r_max:
                r_max = pixel

    # Melakukan contrast stretching
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if r_max == r_min:

                hasil[i, j] = image[i, j]

            else:

                s = ((image[i, j] - r_min) /
                     (r_max - r_min)) * 255

                if s < 0:
                    s = 0

                if s > 255:
                    s = 255

                hasil[i, j] = int(s)

    return hasil

def histogram_equalization(image):

    hist = histogram_manual(image)

    # Jumlah seluruh pixel
    total_pixel = image.shape[0] * image.shape[1]

    cdf = [0] * 256

    cdf[0] = hist[0]

    for i in range(1, 256):

        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = 0

    for i in range(256):

        if cdf[i] != 0:

            cdf_min = cdf[i]

            break

    mapping = [0] * 256

    for i in range(256):

        if total_pixel == cdf_min:

            mapping[i] = 0

        else:

            nilai = ((cdf[i] - cdf_min) /
                     (total_pixel - cdf_min)) * 255

            if nilai < 0:
                nilai = 0

            if nilai > 255:
                nilai = 255

            mapping[i] = int(nilai)

    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            pixel = image[i, j]

            hasil[i, j] = mapping[pixel]

    return hasil

img_negative = negative(img)

img_log = log_transform(img)

# Gamma = 0.5
img_gamma = gamma_transform(img, 0.5)

img_contrast = contrast_stretching(img)

img_equalized = histogram_equalization(img)

hist_original = histogram_manual(img)

hist_negative = histogram_manual(img_negative)

hist_log = histogram_manual(img_log)

hist_gamma = histogram_manual(img_gamma)

hist_contrast = histogram_manual(img_contrast)

hist_equalized = histogram_manual(img_equalized)

plt.figure(figsize=(14, 9))

plt.subplot(2, 3, 1)

img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)

plt.title("Gambar Asli")

plt.axis("off")

plt.subplot(2, 3, 2)

plt.imshow(img_negative, cmap="gray")

plt.title("Negative")

plt.axis("off")

plt.subplot(2, 3, 3)

plt.imshow(img_log, cmap="gray")

plt.title("Log Transform")

plt.axis("off")

plt.subplot(2, 3, 4)

plt.imshow(img_gamma, cmap="gray")

plt.title("Gamma Transform")

plt.axis("off")

plt.subplot(2, 3, 5)

plt.imshow(img_contrast, cmap="gray")

plt.title("Contrast Stretching")

plt.axis("off")

plt.subplot(2, 3, 6)

plt.imshow(img_equalized, cmap="gray")

plt.title("Histogram Equalization")

plt.axis("off")


plt.tight_layout()

plt.show()

plt.figure(figsize=(14, 9))

plt.subplot(2, 3, 1)

plt.bar(
    range(256),
    hist_original,
    width=1
)

plt.title("Histogram Asli")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 2)

plt.bar(
    range(256),
    hist_negative,
    width=1
)

plt.title("Histogram Negative")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 3)

plt.bar(
    range(256),
    hist_log,
    width=1
)

plt.title("Histogram Log")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 4)

plt.bar(
    range(256),
    hist_gamma,
    width=1
)

plt.title("Histogram Gamma")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 5)

plt.bar(
    range(256),
    hist_contrast,
    width=1
)

plt.title("Histogram Contrast Stretching")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")

plt.subplot(2, 3, 6)

plt.bar(
    range(256),
    hist_equalized,
    width=1
)

plt.title("Histogram Equalization")

plt.xlabel("Intensitas")

plt.ylabel("Jumlah Pixel")


plt.tight_layout()

plt.show()