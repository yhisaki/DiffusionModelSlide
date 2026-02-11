import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 1, 500)

beta_min = 0.1
beta_max = 20.0

# beta(s) の積分と alpha_t の計算
integral_beta = beta_min * t + 0.5 * (beta_max - beta_min) * t**2
alpha_t = np.exp(-integral_beta)

x_0 = 1.0
mean = np.sqrt(alpha_t) * x_0
std = np.sqrt(1 - alpha_t)

plt.figure(figsize=(9, 5))

plt.plot(
    t,
    mean,
    label=r"Mean $\mu(t) = \sqrt{\alpha_t} x_0$",
    color="#1f77b4",
    linewidth=2.5,
)

plt.fill_between(
    t,
    mean - std,
    mean + std,
    color="#1f77b4",
    alpha=0.2,
    label=r"$\pm \sigma(t)$ interval",
)
plt.plot(t, mean + std, color="#1f77b4", linestyle="--", alpha=0.5)
plt.plot(t, mean - std, color="#1f77b4", linestyle="--", alpha=0.5)

plt.axhline(0, color="black", linewidth=1, linestyle="-")
plt.xlim(0, 1)
plt.ylim(-2.5, 2.5)
plt.xlabel("Time $t$", fontsize=14)
plt.ylabel("State $x$", fontsize=14)
plt.title("VP SDE: Evolution of Mean and Variance ($x_0 = 1$)", fontsize=16)
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("assets/vp_sde_plot.png", dpi=300)
