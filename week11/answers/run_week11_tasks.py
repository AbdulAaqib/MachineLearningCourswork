import os
import math
from typing import Tuple, List

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../week11
ANS_DIR = os.path.join(BASE_DIR, "answers")
DATA_REGRESSION = os.path.join(BASE_DIR, "Regression_data.csv")
DATA_ADVERT = os.path.join(BASE_DIR, "Regression_Advertising.csv")
ANS_TXT = os.path.join(ANS_DIR, "answers.txt")

# Utility loaders

def load_xy_csv(path: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                # skip header or malformed
                continue
            xs.append(x)
            ys.append(y)
    return xs, ys

# Math helpers (pure Python / no numpy requirement)

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")

def lse_for_model(xs: List[float], ys: List[float], m: float, b: float) -> float:
    # Least Squares Error: sum((m*x + b - y)^2)
    return sum((m * x + b - y) ** 2 for x, y in zip(xs, ys))

def fit_simple_linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    # Returns (slope m, intercept b)
    x_bar = mean(xs)
    y_bar = mean(ys)
    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den = sum((x - x_bar) ** 2 for x in xs)
    m = num / den
    b = y_bar - m * x_bar
    return m, b

def cost_function_J(xs: List[float], ys: List[float], m: float, b: float) -> float:
    # J(θ) = (1/(2m_samples)) * Σ (h_θ(x_i) - y_i)^2
    m_samples = len(xs)
    sse = lse_for_model(xs, ys, m, b)
    return sse / (2.0 * m_samples)

# Optional plotting (attempt matplotlib if available)

def try_plot_advertising(xs: List[float], ys: List[float], m: float, b: float) -> List[str]:
    saved = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Scatter + regression line
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, color="#1f77b4", alpha=0.8, label="Data")
        x_min, x_max = min(xs), max(xs)
        x_line = [x_min, x_max]
        y_line = [m * x_min + b, m * x_max + b]
        ax.plot(x_line, y_line, color="#d62728", linewidth=2, label=f"Fit: y = {m:.4f}x + {b:.2f}")
        ax.set_title("Advertising Spend vs Sales (Linear Regression)")
        ax.set_xlabel("Advertising Spend (£)")
        ax.set_ylabel("Sales (£)")
        ax.legend()
        out1 = os.path.join(ANS_DIR, "task3_advertising_regression.png")
        fig.tight_layout()
        fig.savefig(out1, dpi=160)
        saved.append(out1)
        plt.close(fig)

        # Residuals plot
        residuals = [(m * x + b) - y for x, y in zip(xs, ys)]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(xs, residuals, color="#2ca02c", alpha=0.8)
        ax2.axhline(0.0, color="black", linewidth=1)
        ax2.set_title("Residuals vs Advertising Spend")
        ax2.set_xlabel("Advertising Spend (£)")
        ax2.set_ylabel("Residual (Prediction - Actual)")
        out2 = os.path.join(ANS_DIR, "task3_advertising_residuals.png")
        fig2.tight_layout()
        fig2.savefig(out2, dpi=160)
        saved.append(out2)
        plt.close(fig2)

    except Exception as e:
        # plotting not available
        pass
    return saved


def main():
    os.makedirs(ANS_DIR, exist_ok=True)

    # Load datasets
    xs_reg, ys_reg = load_xy_csv(DATA_REGRESSION)
    xs_adv, ys_adv = load_xy_csv(DATA_ADVERT)

    # Task 1: LSE for model y = 6.49x + 30.18 using Regression_data.csv
    t1_m, t1_b = 6.49, 30.18
    t1_lse = lse_for_model(xs_reg, ys_reg, t1_m, t1_b)

    # Task 2: Evaluate three candidate models against Advertising dataset (used as proxy)
    candidates = [
        (22.60, 169.2),
        (23.42, 167.7),
        (23.10, 168.1),
    ]
    t2_lse_list = []
    for (m, b) in candidates:
        lse_val = lse_for_model(xs_adv, ys_adv, m, b)
        t2_lse_list.append(((m, b), lse_val))
    t2_lse_list.sort(key=lambda kv: kv[1])  # ascending by LSE
    t2_best = t2_lse_list[0]

    # Task 3: Fit linear regression to Advertising dataset
    m_hat, b_hat = fit_simple_linear_regression(xs_adv, ys_adv)
    pred_0 = b_hat
    pred_7500 = m_hat * 7500.0 + b_hat

    # Generate plots
    saved_plots = try_plot_advertising(xs_adv, ys_adv, m_hat, b_hat)

    # Task 4: Compute cost function J for fitted line in Task 3
    J_hat = cost_function_J(xs_adv, ys_adv, m_hat, b_hat)

    # Prepare answers.txt content
    lines = []
    lines.append("Machine Learning - Week 11 Answers (Regression)\n")
    lines.append("\n")
    lines.append("Task 1: Least Squares Error for y = 6.49x + 30.18 (using Regression_data.csv)\n")
    lines.append(f"- LSE = {t1_lse:.6f}\n")
    lines.append("\n")
    lines.append("Task 2: Best Regression Model (by Least Squares Error)\n")
    lines.append("Models evaluated on available advertising dataset (used as a proxy for the Benetton data):\n")
    for (m, b), lse_val in t2_lse_list:
        lines.append(f"- y = {m:.2f}x + {b:.1f} -> LSE = {lse_val:.6f}\n")
    (best_m, best_b), best_lse = t2_best
    lines.append(f"Best (lowest LSE): y = {best_m:.2f}x + {best_b:.1f} (LSE = {best_lse:.6f})\n")
    lines.append("\n")
    lines.append("Task 3: Linear model for Software company (Regression_Advertising.csv)\n")
    lines.append(f"- Fitted line: y = {m_hat:.6f}x + {b_hat:.6f}\n")
    lines.append(f"- Expected sales when budget is £0: {pred_0:.2f}\n")
    lines.append(f"- Expected sales when budget is £7,500: {pred_7500:.2f}\n")
    if saved_plots:
        lines.append("- Plots saved:\n")
        for p in saved_plots:
            relp = os.path.relpath(p, BASE_DIR)
            lines.append(f"  * {relp}\n")
    else:
        lines.append("- Plotting libraries unavailable; no plots generated.\n")
    lines.append("\n")
    lines.append("Task 4: Cost function J for fitted line (Task 3)\n")
    lines.append("We define J(θ) = (1/(2m)) * Σ (h_θ(x_i) - y_i)^2.\n")
    lines.append(f"- J for fitted parameters (θ0 = {b_hat:.6f}, θ1 = {m_hat:.6f}) on advertising data: {J_hat:.6f}\n")
    lines.append("\n")
    lines.append("Code snapshot (additions for cost function):\n")
    lines.append("""\
# -- Begin snapshot --

def cost_function_J(xs: List[float], ys: List[float], m: float, b: float) -> float:
    # J(θ) = (1/(2m_samples)) * Σ (h_θ(x_i) - y_i)^2
    m_samples = len(xs)
    sse = sum((m * x + b - y) ** 2 for x, y in zip(xs, ys))
    return sse / (2.0 * m_samples)

# Usage after fitting (m_hat, b_hat):
J_hat = cost_function_J(xs_adv, ys_adv, m_hat, b_hat)
# -- End snapshot --
""")

    with open(ANS_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Also persist numeric results to a small JSON-like txt for quick checks
    with open(os.path.join(ANS_DIR, "results_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Task1_LSE={t1_lse}\n")
        for idx, ((m, b), l) in enumerate(t2_lse_list, 1):
            f.write(f"Task2_Model{idx}_m={m},b={b},LSE={l}\n")
        f.write(f"Task3_m_hat={m_hat},b_hat={b_hat},pred_0={pred_0},pred_7500={pred_7500}\n")
        f.write(f"Task4_J_hat={J_hat}\n")

if __name__ == "__main__":
    main()
