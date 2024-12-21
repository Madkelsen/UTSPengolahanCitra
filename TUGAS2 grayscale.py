import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar RGB
image_rgb = img.imread("C:/Users/Numotion Editor/Documents/matplotlib_project/source.jpg")

# Konversi ke grayscale
image_gray = np.dot(image_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# Tampilkan gambar grayscale
plt.imshow(image_gray, cmap='gray')
plt.title("Gambar Grayscale")
plt.axis("off")  # Hilangkan sumbu untuk tampilan lebih bersih
plt.show()
