import time, copy, sys, warnings, itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Patch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError): pass
rng = np.random.default_rng(42)
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": .3,
                     "savefig.dpi": 180, "axes.unicode_minus": False})

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
step    = lambda z: (np.asarray(z) >= 0).astype(float)
ACT     = {"logistic": sigmoid, "relu": lambda z: np.maximum(z, 0)}


def metrics(y, p):
    return (accuracy_score(y, p), precision_score(y, p, zero_division=0),
            recall_score(y, p, zero_division=0))


def show(name, tr, te=None):
    a, p, r = tr
    s = f"{name:<34s} train: acc={a:.4f} prec={p:.4f} rec={r:.4f}"
    if te is not None:
        a, p, r = te
        s += f" | test: acc={a:.4f} prec={p:.4f} rec={r:.4f}"
    print(s)


def train_mlp(X, y, hidden, activation="logistic", seed=42, max_iter=2000,
              learning_rate_init=.2, batch_size=64, alpha=0.0,
              tol=1e-7, n_iter_no_change=100):
    m = MLPClassifier(hidden, activation=activation, solver="sgd", momentum=0,
                      learning_rate_init=learning_rate_init, batch_size=batch_size,
                      alpha=alpha, max_iter=max_iter, tol=tol,
                      n_iter_no_change=n_iter_no_change, random_state=seed)
    m.fit(X, y)
    return m


def forward_layers(m, X):
    A = np.asarray(X, float)
    fn = ACT[m.activation]
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        A = fn(A @ W + b)
    return A, A @ m.coefs_[-1] + m.intercepts_[-1]


def step_predict(m, X):
    A = np.asarray(X, float)
    for W, b in zip(m.coefs_[:-1], m.intercepts_[:-1]):
        A = step(A @ W + b)
    return step(A @ m.coefs_[-1] + m.intercepts_[-1]).ravel().astype(int)


def evaluate(model, Xtr, ytr, Xte, yte):
    return metrics(ytr, model.predict(Xtr)) + metrics(yte, model.predict(Xte))


# ------------------------- Задание 1 ---------------------------------------
def in_angle(P):
    x, y = P[:, 0], P[:, 1]
    return (2 * x - y >= 0) & (-.5 * x + y >= 0)


def gen_angle(n=2500, box=5.0):
    classes = []
    for cls, cond in ((1, in_angle), (0, lambda P: ~in_angle(P))):
        parts = []
        while sum(map(len, parts)) < n:
            C = rng.uniform(-box, box, (20000, 2))
            parts.append(C[cond(C)])
        classes.append((np.vstack(parts)[:n], np.full(n, cls)))
    X = np.vstack([c[0] for c in classes])
    Y = np.concatenate([c[1] for c in classes])
    idx = rng.permutation(len(Y))
    return X[idx], Y[idx]


W1 = np.array([[2., -1.], [-.5, 1.]]);  b1 = np.zeros(2)
W2 = np.array([[1., 1.]]);              b2 = np.array([-1.5])


def manual(X, act="step", scale=1.0):
    z1 = X @ (W1 * scale).T + b1
    H = step(z1) if act == "step" else sigmoid(z1)
    z2 = H @ W2.T + b2
    out = step(z2) if act == "step" else (sigmoid(z2) >= .5)
    return H, out.ravel().astype(int)


def plot_angle(X, y):
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for c, col, lab in ((0, "tab:blue", "класс 0"), (1, "tab:red", "класс 1")):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=5, alpha=.35, c=col,
                   label=lab, edgecolors="none")
    xx = np.linspace(-5, 5, 300)
    ax.plot(xx, 2 * xx, "k--", lw=1.5, label=r"$L_1:2x-y=0$")
    ax.plot(xx, .5 * xx, "k-.", lw=1.5, label=r"$L_2:-0.5x+y=0$")
    ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel="$x_1$", ylabel="$x_2$",
           title="Задание 1. Острый угол (тип 5)")
    ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig1_angle_data.png"); plt.close(fig)


def plot_angle_acts(m):
    g = np.linspace(-5, 5, 350); xx, yy = np.meshgrid(g, g)
    P = np.column_stack([xx.ravel(), yy.ravel()])
    Hm, om = manual(P)
    Ht, logit = forward_layers(m, P)
    pt = sigmoid(logit).ravel()
    panels = [(Hm[:, 0].reshape(xx.shape), r"$h_1=S(2x_1-x_2)$"),
              (Hm[:, 1].reshape(xx.shape), r"$h_2=S(-0.5x_1+x_2)$"),
              (om.reshape(xx.shape),           "выход, ступенчатая"),
              (Ht[:, 0].reshape(xx.shape),     "обученный нейрон 1"),
              (Ht[:, 1].reshape(xx.shape),     "обученный нейрон 2"),
              (pt.reshape(xx.shape),           r"выход, $P(y=1)$")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for ax, (Z, title) in zip(axes.ravel(), panels):
        im = ax.imshow(Z, extent=(-5, 5, -5, 5), origin="lower",
                       cmap="RdYlBu_r" if "выход" in title else "viridis",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$"); ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.suptitle("Активации скрытых нейронов: аналитическая (сверху) "
                 "и обученная (снизу) сети")
    fig.tight_layout(); fig.savefig(FIG / "fig2_angle_activations.png"); plt.close(fig)


def plot_sweep_prune(ns, acc_tr, acc_te, pr_te, rec_te, labels, pm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(ns, acc_tr[:, 0], yerr=acc_tr[:, 1], marker="o", capsize=3, label="train")
    axes[0].errorbar(ns, acc_te[:, 0], yerr=acc_te[:, 1], marker="s", capsize=3, label="test")
    axes[0].set(xlabel="нейронов", ylabel="accuracy", ylim=(.82, 1.005),
                title="Accuracy от числа нейронов"); axes[0].legend()
    axes[1].errorbar(ns, pr_te[:, 0],  yerr=pr_te[:, 1],  marker="^", capsize=3, label="precision")
    axes[1].errorbar(ns, rec_te[:, 0], yerr=rec_te[:, 1], marker="d", capsize=3, label="recall")
    axes[1].set(xlabel="нейронов", ylabel="метрика", ylim=(.82, 1.005),
                title="Precision и recall (test)"); axes[1].legend()
    pos = np.arange(len(labels))
    for k, (name, col) in enumerate((("accuracy", "tab:blue"),
                                     ("precision", "tab:orange"),
                                     ("recall", "tab:green"))):
        axes[2].bar(pos + (k - 1) * .27, [r[name] for r in pm], .27,
                    label=name, color=col)
    axes[2].set_xticks(pos, labels, rotation=20, ha="right")
    axes[2].set(ylim=(0, 1.02), ylabel="метрика",
                title="Обнуление случайных нейронов")
    axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig3_angle_neurons_pruning.png"); plt.close(fig)


def task1():
    print("\n" + "=" * 78 + "\nЗАДАНИЕ 1. Острый угол\n" + "=" * 78)
    X, y = gen_angle()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3,
                                          random_state=42, stratify=y)
    print(f"Всего {len(y)}, train {len(ytr)}, test {len(yte)}")
    plot_angle(X, y)

    p_step_te = manual(Xte)[1]
    show("Аналитика, ступенчатая",
         metrics(ytr, manual(Xtr)[1]), metrics(yte, p_step_te))
    for name, scale in (("Аналитика, сигмоида (W1)", 1.0),
                        ("Аналитика, сигмоида (10·W1)", 10.0)):
        show(name,
             metrics(ytr, manual(Xtr, "sigmoid", scale)[1]),
             metrics(yte, manual(Xte, "sigmoid", scale)[1]))

    m = train_mlp(Xtr, ytr, (2,), max_iter=4000)
    show("Обученная сеть 2-1",
         metrics(ytr, m.predict(Xtr)), metrics(yte, m.predict(Xte)))
    show("Те же веса, ступеньки",
         metrics(ytr, step_predict(m, Xtr)), metrics(yte, step_predict(m, Xte)))
    print("Выученные веса скрытого слоя (столбцы coefs_[0]):")
    for i in range(2):
        w, b = m.coefs_[0][:, i], m.intercepts_[0][i]
        wn = w / np.linalg.norm(w)
        for name, a in (("L1", np.array([2., -1.])), ("L2", np.array([-.5, 1.]))):
            a = a / np.linalg.norm(a)
            print(f"  нейрон {i+1}: w={np.round(w,3)}, b={b:+.3f}; "
                  f"|cos| с {name} = {abs(wn @ a):.6f}")
    print(f"Согласие с аналитикой на тесте: "
          f"{np.mean(p_step_te == m.predict(Xte)):.4f}")
    plot_angle_acts(m)

    print("\nЧисло нейронов: 1, 2, 3 (3 инициализации):")
    ns = [1, 2, 3];  sweep = []
    for n in ns:
        v = np.array([evaluate(train_mlp(Xtr, ytr, (n,), seed=s, max_iter=3000,
                                          n_iter_no_change=300),
                               Xtr, ytr, Xte, yte) for s in (0, 1, 2)])
        sweep.append(v)
        print(f"  n={n}: train acc={v[:,0].mean():.4f}±{v[:,0].std():.4f}, "
              f"test acc={v[:,3].mean():.4f}±{v[:,3].std():.4f}, "
              f"prec={v[:,4].mean():.4f}, rec={v[:,5].mean():.4f}")
    A = np.array(sweep)
    ms = lambda c: np.column_stack([A[:, :, c].mean(1), A[:, :, c].std(1)])
    acc_tr, acc_te, pr_te, rec_te = ms(0), ms(3), ms(4), ms(5)

    print("\nОбнуление случайных нейронов (16 нейронов):")
    mp = train_mlp(Xtr, ytr, (16,), max_iter=4000)
    rand = np.random.default_rng(42).permutation(16)
    specs = [("0%", []), ("25% случайных", rand[:4]), ("50% случайных", rand[:8])]
    pm = []
    for name, idx in specs:
        mm = copy.deepcopy(mp)
        if len(idx):
            mm.coefs_[0][:, idx] = 0
            mm.intercepts_[0][idx] = 0
            mm.coefs_[1][idx, :] = 0
        pm.append(dict(zip(("accuracy", "precision", "recall"),
                           metrics(yte, mm.predict(Xte)))))
        show(name, metrics(ytr, mm.predict(Xtr)), metrics(yte, mm.predict(Xte)))
    plot_sweep_prune(ns, acc_tr, acc_te, pr_te, rec_te,
                     [s[0] for s in specs], pm)


# ------------------------- Задание 2 ---------------------------------------
def sample(n, cond, box=2.0, seed=42):
    r = np.random.default_rng(seed); parts = []
    while sum(map(len, parts)) < n:
        C = r.uniform(-box, box, (50000, 3))
        parts.append(C[cond(C)])
    return np.vstack(parts)[:n]


c4 = lambda P: P[:, 0] + P[:, 1] + P[:, 2] > 3
c5 = lambda P: P[:, 0] - P[:, 1] + P[:, 2] < -3
c7 = lambda P: ~((np.abs(P[:, 0] + 1) <= .25) & (np.abs(P[:, 1] - 2) <= .5)
                 & (np.abs(P[:, 2] - 1) <= .75))


def gen3d():
    P4, P5, P7 = sample(200, c4, seed=101), sample(200, c5, seed=102), sample(400, c7, seed=103)
    X = np.vstack([P4, P5, P7])
    y = np.r_[np.ones(400, int), np.zeros(400, int)]
    idx = np.random.default_rng(42).permutation(len(y))
    return X[idx], y[idx], P4, P5, P7


def draw_box(ax):
    c, h = np.array([-1., 2., 1.]), np.array([.25, .5, .75])
    V = c + np.array(list(itertools.product((-1, 1), repeat=3))) * h
    E = [[V[i], V[j]] for i in range(8) for j in range(i + 1, 8)
         if np.sum(np.abs(V[i] - V[j]) > 1e-9) == 1]
    ax.add_collection3d(Line3DCollection(E, colors="k", lw=1.2))


def plot3d(P4, P5, P7):
    fig = plt.figure(figsize=(7.8, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(-2, 2, 70); X, Y = np.meshgrid(u, u)
    for Z, color in ((3 - X - Y, "tab:red"), (-3 - X + Y, "tab:orange")):
        Z = np.ma.masked_where((Z < -2) | (Z > 2), Z)
        ax.plot_surface(X, Y, Z, color=color, alpha=.14,
                        linewidth=0, shade=False, antialiased=True)
    for P, c, m, lab, size in [(P4, "tab:red", "^", "тип 4", 14),
                               (P5, "tab:orange", "v", "тип 5", 14),
                               (P7, "tab:blue", "o", "тип 7", 9)]:
        ax.scatter(*P.T, s=size, c=c, marker=m, alpha=.85,
                   label=lab, edgecolors="none")
    draw_box(ax)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2),
           xlabel="x", ylabel="y", zlabel="z", title="Типы точек 4, 5, 7")
    ax.view_init(22, 38)
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="tab:red",    alpha=.3, label="плоскость $x+y+z=3$"),
                Patch(facecolor="tab:orange", alpha=.3, label="плоскость $x-y+z=-3$")]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig4_3d_data.png"); plt.close(fig)


def train3d(X, y, hidden):
    return train_mlp(X, y, hidden, "relu", max_iter=300, learning_rate_init=.1,
                     batch_size=32, alpha=1e-4, tol=0, n_iter_no_change=1000)


def time3d(X, y, hidden, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.process_time()
        train3d(X, y, hidden)
        times.append(time.process_time() - t0)
    return float(np.median(times))


def plot_width(ns, tt, tr, te):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes[0, 0].plot(ns, tt, "s-")
    axes[0, 0].set_title("Полное время обучения")
    axes[0, 0].set_ylabel("с")
    for (title, i, ylim), ax in zip(
            [("Accuracy", 0, (.85, 1)), ("Precision", 1, (.80, 1)),
             ("Recall",   2, (.80, 1.01))],
            [axes[0, 1], axes[1, 0], axes[1, 1]]):
        ax.plot(ns, [m[i] for m in tr], "o-", label="train")
        ax.plot(ns, [m[i] for m in te], "s-", label="test")
        ax.set(title=title, ylim=ylim); ax.legend()
    for ax in axes.ravel():
        ax.set_xlabel("нейронов в первом скрытом слое")
    fig.suptitle("Задание 2. Влияние числа нейронов")
    fig.tight_layout(); fig.savefig(FIG / "fig5_3d_width.png"); plt.close(fig)


def task2():
    print("\n" + "=" * 78 + "\nЗАДАНИЕ 2. Типы 4, 5, 7\n" + "=" * 78)
    X, y, P4, P5, P7 = gen3d()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3,
                                          random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    print(f"train {len(ytr)}, test {len(yte)}; классы: "
          f"{np.bincount(ytr)}, {np.bincount(yte)}")
    plot3d(P4, P5, P7)

    ns = list(range(5, 51, 5)); rows = []
    train_mlp(Xtr, ytr, (5, 10), activation="relu", seed=0, max_iter=5)
    print("\n10 сетей (n,10), ReLU, 300 эпох:")
    for n in ns:
        m = train3d(Xtr, ytr, (n, 10))
        a_tr = metrics(ytr, m.predict(Xtr))
        a_te = metrics(yte, m.predict(Xte))
        tt = time3d(Xtr, ytr, (n, 10))
        rows.append((n, tt, a_tr, a_te))
        print(f"  n={n:2d}: t_total={tt:.4f}, "
              f"train acc={a_tr[0]:.4f} prec={a_tr[1]:.4f} rec={a_tr[2]:.4f}, "
              f"test acc={a_te[0]:.4f} prec={a_te[1]:.4f} rec={a_te[2]:.4f}")
    plot_width(ns, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])


task1()
task2()
print("\nГотово. Рисунки в", FIG)
