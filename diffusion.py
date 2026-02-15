import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import numpy as np

# 1. データ生成（Swiss Roll）
n_samples = 2000
data, _ = make_swiss_roll(n_samples=n_samples, noise=0.0)
# 2次元に投影（X軸とZ軸を使用）し、スケーリング
x = data[:, [0, 2]] / 10.0

# 2. ノイズレベルの設定（t=0, t=T/2, t=T を模倣）
noise_levels = [0.0, 0.25, 0.5, 2.0]  # 左から右へノイズを強くする
titles = [r"$t=0$ (Data)", r"$t \approx T/2$ (Diffusing)", r"$t=T$ (Noise)"]

# 3. プロット描画
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    # ノイズを加える（正規分布）
    noise = np.random.normal(scale=noise_levels[i], size=x.shape)
    x_noisy = x + noise

    # 散布図
    ax.scatter(
        x_noisy[:, 0], x_noisy[:, 1], s=5, alpha=0.6, c="#1f77b4", edgecolors="none"
    )

    # 見た目の調整
    ax.set_title(titles[i], fontsize=16)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # 枠線を消してスッキリさせる（任意）
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#dddddd")

plt.tight_layout()
plt.show()
