# Nama : Armada Satya Permana
# NIM : 220401010131
# Kelas : IFD51
# Mata Kuliah : Pengolahan Citra

import imageio
import numpy as np
import matplotlib.pyplot as plt

image = imageio.v2.imread("C:/Users/Numotion Editor/Documents/matplotlib_project/lowcontrastpicture.png")

if len(image.shape) == 3:
    gray_image = np.dot(image[...,:3], [0.299, 0.587, 0.114]) 
else:
    gray_image = image

contrast_level = 1.5
enhanced_image = np.clip(contrast_level * gray_image, 0, 255).astype('uint8')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Citra Asli")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Kontras Level 1.5")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
equalized_image = cdf_final[gray_image.astype('uint8')]

plt.subplot(1, 3, 3)
plt.title("Histogram Equalization")
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')

plt.show()