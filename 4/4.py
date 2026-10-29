import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import rand_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path(__file__).resolve().parent

# ------------------------- Задание 2: числовые данные -------------------------
def make_type2(n, rng):
    # (a, b + 0.5, a * eps), a, b, eps ~ U[-1, 1]
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(-1, 1, n)
    eps = rng.uniform(-1, 1, n)
    return np.column_stack([a, b + 0.5, a * eps])


def make_type4(n, rng):
    # (a, b, -0.5 + b * eps), a, b, eps ~ N(0, 0.5^2)
    a = rng.normal(0, 0.5, n)
    b = rng.normal(0, 0.5, n)
    eps = rng.normal(0, 0.5, n)
    return np.column_stack([a, b, -0.5 + b * eps])


def make_points(seed, n=300):
    rng = np.random.default_rng(seed)
    X = np.vstack([make_type2(n, rng), make_type4(n, rng)])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    order = rng.permutation(len(y))
    return X[order], y[order]


def birch_result(X, n_clusters):
    X_scaled = StandardScaler().fit_transform(X)
    labels = Birch(n_clusters=n_clusters, threshold=0.1).fit_predict(X_scaled)
    return labels, X_scaled


print("\nЗадание 2. Числовые данные")
print("k  Rand index          Davies-Bouldin")
for k in [2, 3, 4, 5, 6]:
    rand_values = []
    db_values = []
    for seed in range(5):
        X, y = make_points(seed)
        labels, X_scaled = birch_result(X, k)
        rand_values.append(rand_score(y, labels))
        db_values.append(davies_bouldin_score(X_scaled, labels))
    print(f"{k}  {np.mean(rand_values):.3f} ± {np.std(rand_values):.3f}       "
          f"{np.mean(db_values):.3f} ± {np.std(db_values):.3f}")

X, y = make_points(0)
fig = plt.figure(figsize=(12, 7.5))

ax = fig.add_subplot(2, 3, 1, projection="3d")
for value, color in [(0, "tab:blue"), (1, "tab:orange")]:
    ax.scatter(X[y == value, 0], X[y == value, 1], X[y == value, 2],
               s=8, alpha=0.7, color=color, label=f"группа {value}")
ax.set_title("Истинные группы")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend(fontsize=7)

for i, k in enumerate([2, 3, 4, 5, 6], start=2):
    labels, _ = birch_result(X, k)
    ax = fig.add_subplot(2, 3, i, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="tab10",
               vmin=0, vmax=max(1, k - 1), s=8, alpha=0.7)
    ax.set_title(f"BIRCH, k={k}")
    ax.text2D(0.02, 0.95, f"Rand = {rand_score(y, labels):.3f}",
              transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

fig.suptitle("Числовые данные: истинные группы и кластеры BIRCH (запуск 0)", fontsize=12)
fig.tight_layout()
fig.savefig(BASE / "fig1_numeric_clusters.png", dpi=150)
plt.close(fig)

# ------------------------- Задания 3 и 4: тексты -------------------------
print("\nЗадание 3. Текстовая выборка")
files = sorted((BASE / "recipes").glob("*.txt"))
texts = [p.read_text(encoding="utf-8") for p in files]
y_text = np.array([0 if p.stem.startswith("dessert") else 1 for p in files])
print(f"Всего текстов: {len(texts)}, десертов: {sum(y_text == 0)}, "
      f"закусок/мясных и рыбных блюд: {sum(y_text == 1)}")


def cluster_texts(corpus, method):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    if method == "BIRCH":
        model = Birch(n_clusters=2, threshold=0.1)
    else:
        model = KMeans(n_clusters=2, n_init=10, random_state=0)
    return model.fit_predict(X), X


def show_text_result(name, labels, X):
    rand = rand_score(y_text, labels)
    db = davies_bouldin_score(X.toarray(), labels)
    print(f"{name}: Rand index = {rand:.3f}, Davies-Bouldin = {db:.3f}")


print("\nЗадание 4. Кластеризация исходных текстов")
for method in ["BIRCH", "KMeans"]:
    labels, X = cluster_texts(texts, method)
    show_text_result(method, labels, X)


def add_common_words(corpus, labels):
    result = []
    for text, label in zip(corpus, labels):
        if label == 0:
            result.append(text + " sweet dessert sugar bake oven cake chocolate")
        else:
            result.append(text + " savory meat fish garlic onion salt pepper")
    return result


texts_improved = add_common_words(texts, y_text)

print("\nПосле добавления общих слов в тексты каждой группы")
for method in ["BIRCH", "KMeans"]:
    labels, X = cluster_texts(texts_improved, method)
    show_text_result(method, labels, X)

print("\nГотово. Рисунок сохранён в", BASE / "fig1_numeric_clusters.png")
