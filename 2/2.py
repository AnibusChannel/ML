import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Параметры
N = 200

# 2. Группа 1: тип 2
def make_group_type2(n):
    a = np.random.uniform(-1, 1, n)
    b = np.random.uniform(-1, 1, n)
    eps = np.random.uniform(-1, 1, n)
    return np.column_stack([a, b - 0.5, a * eps])

# 3. Группа 2: тип 5
def make_group_type5(n):
    a = np.random.normal(0, 0.5, n)
    b = np.random.normal(0, 0.5, n)
    eps = np.random.normal(0, 0.5, n)
    return np.column_stack([0.1 * a - 1, 3 * b, eps - 1])

# 4. Группа 3: тип 8
def make_group_type8(n):
    r = np.random.uniform(0, 1, n) ** (1 / 3)
    cos_theta = np.random.uniform(0, 1, n)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    phi = np.random.uniform(0, 2 * np.pi, n)
    return np.column_stack([
        r * sin_theta * np.cos(phi),
        r * sin_theta * np.sin(phi),
        r * cos_theta
    ])

# 5. Собираем три группы
X1 = make_group_type2(N)
X2 = make_group_type5(N)
X3 = make_group_type8(N)
X = np.vstack([X1, X2, X3])

# 6. Нормализация
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 7. Три схемы разметки
y_1vs23 = np.array([0] * N + [1] * (2 * N))
y_12vs3 = np.array([0] * (2 * N) + [1] * N)
y_multi = np.array([0] * N + [1] * N + [2] * N)

# 8. Модели
def get_models():
    return {
        "Линейный порог (SGD)": SGDClassifier(
            loss='perceptron', max_iter=1000, eta0=0.1
        ),
        "Логистическая регрессия": LogisticRegression(
            C=1.0, max_iter=1000
        ),
        "Наивный Байес": GaussianNB(),
    }

# 9. Оценка
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)
    return acc, prec, rec

# 10. Прогон трёх схем
schemes = {
    "1 vs 2+3": y_1vs23,
    "1+2 vs 3": y_12vs3,
    "3 класса": y_multi,
}

for scheme_name, y in schemes.items():
    print("=" * 60)
    print("Схема:", scheme_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y
    )
    for model_name, model in get_models().items():
        acc, prec, rec = evaluate_model(model, X_train, y_train, X_test, y_test)
        print(f"\n{model_name}: accuracy={acc:.3f}")
        print("precision по классам:", np.round(prec, 3))
        print("recall по классам:   ", np.round(rec, 3))

# 11. Задание 4
X1_new = make_group_type2(N)
X2_new = make_group_type5(N)
X3_new = make_group_type8(N)
X_new = np.vstack([X1_new, X2_new, X3_new])
X_new = scaler.transform(X_new)
y_new = np.array([0] * N + [1] * (2 * N))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_1vs23, test_size=0.3, stratify=y_1vs23
)

print("\n" + "=" * 60)
print("Задание 4: схема 1 vs 2+3")
for model_name, model in get_models().items():
    model.fit(X_train, y_train)
    acc_old = accuracy_score(y_test, model.predict(X_test))
    acc_new = accuracy_score(y_new, model.predict(X_new))
    print(f"{model_name}: старая = {acc_old:.3f}, новая = {acc_new:.3f}")
