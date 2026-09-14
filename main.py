import cv2
import matplotlib.pyplot as plt
import numpy as np

R = 255
G = 255
B = 255

gambar = cv2.imread("tower2.jpg")
gambar = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

red = gambar[:, :, 0]
green = gambar[:, :, 1]
blue = gambar[:, :, 2]

output = np.zeros_like(gambar)

output[:, :, 0] = red * (R / 255)
output[:, :, 1] = green * (G / 255)
output[:, :, 2] = blue * (B / 255)

output = output.astype(np.uint8)

plt.imshow(output)
plt.axis("off")
plt.show()