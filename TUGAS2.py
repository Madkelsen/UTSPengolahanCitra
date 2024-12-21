import imageio as img
import numpy as np
import matplotlib.pyplot as plt

image_rgb = img.imread("C:/Users/Numotion Editor/Documents/matplotlib_project/source.jpg")

image_gray = np.dot(image_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

histogram, bins = np.histogram(image_gray.flatten(), bins=256, range=(0, 255))

plt.figure(figsize=(10, 6))
plt.bar(range(256), histogram, color='gray', width=1)
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas (0-255)")
plt.ylabel("Jumlah Piksel")
plt.show()

for intensity, count in enumerate(histogram):
    print(f"Intensitas {intensity}: {count} piksel")


