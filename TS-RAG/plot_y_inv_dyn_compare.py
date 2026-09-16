import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for dataset in ["ETTh2", "ETTm1"]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for row, cfg in enumerate(["v3", "ord01"]):
        data = np.load(f"y_inv_dyn_exports/{cfg}_{dataset}.npz")
        for col in range(2):
            ax = axes[row, col]
            ax.plot(data["target"][col], label="target", color="black", linewidth=2)
            ax.plot(data["pred"][col], label="final_pred", color="green", linestyle="--")
            ax.plot(data["y_inv"][col], label="y_inv", color="blue")
            ax.plot(data["y_dyn"][col], label="y_dyn", color="red")
            ax.set_title(f"{cfg} - {dataset} sample{col}")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"y_inv_dyn_exports/compare_{dataset}.png", dpi=120)
    print(f"已保存 compare_{dataset}.png")
