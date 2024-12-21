import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Contoh data (ganti dengan data saham sesungguhnya)
# X adalah fitur (misal harga penutupan hari sebelumnya, volume perdagangan, dll.)
# Y adalah harga saham berikutnya (target)
X = np.array([[1], [2], [3], [4], [5]])  # Fitur (misalnya harga penutupan hari sebelumnya)
Y = np.array([2, 3, 5, 7, 8])  # Target (harga saham berikutnya)

# Membagi data menjadi data latih dan uji
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()
model.fit(X_train, Y_train)

# Prediksi harga saham menggunakan model
Y_pred = model.predict(X_test)

# Menghitung MSE
mse = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {mse}')

# Menampilkan grafik prediksi vs nilai aktual
plt.scatter(X_test, Y_test, color='blue', label='Data Aktual')  # Titik data aktual
plt.plot(X_test, Y_pred, color='red', label='Prediksi Regresi Linear')  # Garis prediksi regresi
plt.xlabel('Harga Penutupan Hari Sebelumnya')  # Label sumbu X
plt.ylabel('Harga Saham Berikutnya')  # Label sumbu Y
plt.title('Grafik Prediksi vs Nilai Aktual Regresi Linear')  # Judul grafik
plt.legend()
plt.show()
