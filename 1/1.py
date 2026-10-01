import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

# Генерация данных
n_samples = 150

def generate_data(n):
    x = np.random.normal(3.0, 1.0, size=(n, 1))
    y = x**2
    return x, y

X_train, y_train = generate_data(n_samples)

# График исходных данных
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, alpha=0.6, color='blue')
plt.title("Исходная выборка")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Полиномиальная регрессия 2-й степени
# Матрица признаков: [1, x, x^2]
X_poly_train = np.hstack([np.ones((n_samples, 1)), X_train, X_train**2])

# Нахождение весов по формуле МНК: w = (X^T * X)^(-1) * X^T * y
weights = np.linalg.inv(X_poly_train.T @ X_poly_train) @ X_poly_train.T @ y_train

def predict_poly(X, w):
    X_p = np.hstack([np.ones((len(X), 1)), X, X**2])
    return X_p @ w

y_pred_poly_train = predict_poly(X_train, weights)

# Ядерное сглаживание
kr = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.1)
kr.fit(X_train, y_train)

y_pred_kr_train = kr.predict(X_train)

# Оценка на новой выборке
X_test, y_test = generate_data(n_samples)

y_pred_poly_test = predict_poly(X_test, weights)
y_pred_kr_test = kr.predict(X_test)

# Расчет MSE
mse_poly_tr = mean_squared_error(y_train, y_pred_poly_train)
mse_kr_tr = mean_squared_error(y_train, y_pred_kr_train)

mse_poly_te = mean_squared_error(y_test, y_pred_poly_test)
mse_kr_te = mean_squared_error(y_test, y_pred_kr_test)

print("Обучающая выборка")
print(f"Полином (степень 2): {mse_poly_tr:.8f}")
print(f"Kernel Ridge:        {mse_kr_tr:.8f}\n")

print("Новая выборка")
print(f"Полином (степень 2): {mse_poly_te:.8f}")
print(f"Kernel Ridge:        {mse_kr_te:.8f}")
