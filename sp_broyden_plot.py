"""
SP-Broyden: данные и графики для тезисов конференции МФТИ.
Запуск: python sp_broyden_plot.py
Результат: fig_tezisy.png
"""

import numpy as np
from numpy.linalg import norm, solve, cond
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════
#  Единый солвер SP-Broyden
# ═══════════════════════════════════════════

def sp_broyden_solve(F, x0, p_max=0, reset=False,
                     cond_thresh=1e12, maxiter=500, tol=1e-10):
    """
    Параметры:
      p_max=0,  reset=False  →  классический Бройден
      p_max>0,  reset=False  →  SP-Broyden (секанто-сохраняющий)
      p_max=m,  reset=True   →  Андерсон (мультисекущая пересборка)
    Возвращает: list of (iteration, f_evals, ||F||)
    """
    n = len(x0)
    x = x0.copy().astype(float)
    Fx = F(x)
    f_evals = 1
    hist = [(0, 1, float(norm(Fx)))]

    B = np.eye(n)
    S_hist, Y_hist = [], []

    for k in range(maxiter):
        if norm(Fx) < tol:
            break

        try:
            d = solve(B, -Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(d)):
            break

        x_new = x + d
        Fx_new = F(x_new)
        f_evals += 1
        if not np.all(np.isfinite(Fx_new)):
            break

        y = Fx_new - Fx
        s = d
        S_hist.append(s.copy())
        Y_hist.append(y.copy())

        if reset:
            # ── Ветка Андерсона: пересборка B = I + (Y-S)(S^T S)^{-1} S^T ──
            m = min(p_max, len(S_hist))
            B = np.eye(n)
            if m > 0:
                Sm = np.column_stack(S_hist[-m:])
                Ym = np.column_stack(Y_hist[-m:])
                G = Sm.T @ Sm
                while m > 0 and cond(G) > cond_thresh:
                    m -= 1
                    if m == 0:
                        break
                    Sm = np.column_stack(S_hist[-m:])
                    Ym = np.column_stack(Y_hist[-m:])
                    G = Sm.T @ Sm
                if m > 0:
                    try:
                        B = np.eye(n) + (Ym - Sm) @ solve(G, Sm.T)
                    except np.linalg.LinAlgError:
                        B = np.eye(n)
        else:
            # ── Ветка Бройдена: ранг-1 с выбором v ──
            Bs = B @ s
            p = 0
            if p_max > 0 and len(S_hist) >= 2:
                for p_try in range(1, min(p_max, len(S_hist) - 1) + 1):
                    cols = [S_hist[-1 - j] for j in range(p_try + 1)]
                    Sp = np.column_stack(cols)
                    G = Sp.T @ Sp
                    if cond(G) < cond_thresh:
                        p = p_try
                    else:
                        break

            if p == 0:
                v = s  # классический Бройден
            else:
                cols = [S_hist[-1 - j] for j in range(p + 1)]
                Sp = np.column_stack(cols)
                G = Sp.T @ Sp
                e1 = np.zeros(p + 1)
                e1[0] = 1.0
                try:
                    v = Sp @ solve(G, e1)
                except np.linalg.LinAlgError:
                    v = s

            denom = v @ s
            if abs(denom) < 1e-15:
                v = s
                denom = s @ s
            if abs(denom) < 1e-15:
                break

            B = B + np.outer(y - Bs, v) / denom

        x = x_new
        Fx = Fx_new
        hist.append((k + 1, f_evals, float(norm(Fx))))

        max_hist = max(p_max + 5, 15)
        if len(S_hist) > max_hist:
            S_hist.pop(0)
            Y_hist.pop(0)

    return hist


# ═══════════════════════════════════════════
#  Тестовые задачи
# ═══════════════════════════════════════════

def discrete_bvp(x):
    n = len(x)
    h = 1.0 / (n + 1)
    r = np.zeros(n)
    for i in range(n):
        ti = (i + 1) * h
        xm = x[i - 1] if i > 0 else 0.0
        xp = x[i + 1] if i < n - 1 else 0.0
        r[i] = 2 * x[i] - xm - xp + h**2 * (x[i] + ti + 1)**3 / 2
    return r

def discrete_bvp_x0(n):
    h = 1.0 / (n + 1)
    return np.array([i * h * (i * h - 1) for i in range(1, n + 1)])

def broyden_banded(x):
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        ji = [j for j in range(max(0, i - 5), min(n, i + 2)) if j != i]
        r[i] = x[i] * (2 + 5 * x[i]**2) + 1 - sum(x[j] * (1 + x[j]) for j in ji)
    return r

def broyden_banded_x0(n):
    return -np.ones(n)


# ═══════════════════════════════════════════
#  Методы для сравнения
# ═══════════════════════════════════════════

METHODS = [
    # (label, p_max, reset, color, linestyle, linewidth)
    ("Бройден",          0,  False, "#888888", (0, (4, 3)),  1.3),
    ("Андерсон (m=10)",  10, True,  "#2060B0", (0, (5, 2)),  1.5),
    ("SP-Broyden (p≤10)", 10, False, "#D03030", "-",         2.3),
]

PROBLEMS = [
    ("Discrete BVP, n = 20",   discrete_bvp,    discrete_bvp_x0(20)),
    ("Broyden Banded, n = 20", broyden_banded,  broyden_banded_x0(20)),
]


# ═══════════════════════════════════════════
#  Запуск и печать результатов
# ═══════════════════════════════════════════

print("=" * 70)
print("  SP-Broyden: данные для тезисов  |  B_0 = I  |  tol = 1e-10")
print("=" * 70)

all_data = {}

for prob_name, F, x0 in PROBLEMS:
    print(f"\n  {prob_name}")
    prob_data = {}
    for label, pm, rst, *_ in METHODS:
        h = sp_broyden_solve(F, x0.copy(), p_max=pm, reset=rst)
        final_res = h[-1][2]
        n_iter = h[-1][0]
        conv = final_res < 1e-10
        tag = "сошелся" if conv else "расходится"
        print(f"    {label:<22s}  iter={n_iter:<4d}  ||F||={final_res:.2e}  [{tag}]")
        prob_data[label] = h
    all_data[prob_name] = prob_data


# ═══════════════════════════════════════════
#  Построение графиков
# ═══════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.2, 2.4))

for ax, (prob_name, F, x0) in zip([ax1, ax2], PROBLEMS):
    data = all_data[prob_name]

    for label, pm, rst, color, ls, lw in METHODS:
        h = data[label]
        iters = [t[0] for t in h]
        resid = [t[2] for t in h]

        # Фильтруем inf/nan
        xs, ys = [], []
        for i, r in zip(iters, resid):
            if np.isfinite(r) and r > 0:
                xs.append(i)
                ys.append(r)
            else:
                break

        if len(xs) < 2:
            continue

        # Метка с числом итераций при сходимости
        tag = ""
        if ys[-1] < 1e-10:
            tag = f" ({xs[-1]})"

        ax.semilogy(xs, ys, color=color, linestyle=ls,
                    linewidth=lw, label=f"{label}{tag}", alpha=0.9)

    # Линия допуска
    ax.axhline(1e-10, color="#00BBCC", lw=0.5, ls=":", alpha=0.4)

    ax.set_title(prob_name, fontsize=8, pad=4)
    ax.set_xlabel("итерация k", fontsize=7)
    ax.set_ylabel("‖F(x)‖", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)

    # ── Обрезка сверху: показываем только зону сходимости ──
    ax.set_ylim(bottom=1e-14, top=1e2)

ax2.legend(fontsize=5.5, loc="upper right", framealpha=0.8,
           borderpad=0.3, handlelength=1.5, labelspacing=0.2)

fig.tight_layout(pad=0.5)
fig.savefig("fig_tezisy.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"\nГрафик сохранен: fig_tezisy.png")
plt.close()
