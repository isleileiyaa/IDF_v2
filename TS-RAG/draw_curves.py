import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 加载数据
# ==========================================
file_rag_prefix = "my_results/ETTh1_retrieve_ETTh1_512_only_self_train_None"
file_base = "my_results/ETTh1_y_pred_base.npy"

y_true = np.load(f"{file_rag_prefix}_y_true.npy")
y_pred_rag = np.load(f"{file_rag_prefix}_y_pred_rag.npy")
y_pred_base = np.load(file_base)

# 去除 batch 维度 (1, N, 64) -> (N, 64)
y_true = np.squeeze(y_true, axis=0)
y_pred_rag = np.squeeze(y_pred_rag, axis=0)
y_pred_base = np.squeeze(y_pred_base, axis=0)

# 取最后一个特征列
if len(y_true.shape) == 3:
    y_true = y_true[:, :, -1]
    y_pred_rag = y_pred_rag[:, :, -1]
    y_pred_base = y_pred_base[:, :, -1]

# ==========================================
# 2. 核心操作：挑一个代表性的预测窗口来看曲线
# 绝不能对所有样本取平均，一平均就全是直线了！
# 我们提取第 0 个测试样本的 64 步预测曲线来看细节。
# (如果你觉得第 0 个不够典型，可以改这个索引，比如 100, 500)
# ==========================================
SAMPLE_IDX = 0  # 🌟 选取第 0 个测试窗口
seq_true = y_true[SAMPLE_IDX, :]
seq_base = y_pred_base[SAMPLE_IDX, :]
seq_rag = y_pred_rag[SAMPLE_IDX, :]

H = len(seq_true) # 预测步长，即 64

# ==========================================
# 3. 计算一阶和二阶差分 (针对这条特定的曲线)
# ==========================================
# 一阶差分 (长度 63)
diff1_true = np.diff(seq_true, n=1)
diff1_base = np.diff(seq_base, n=1)
diff1_rag = np.diff(seq_rag, n=1)

# 二阶差分 (长度 62)
diff2_true = np.diff(seq_true, n=2)
diff2_base = np.diff(seq_base, n=2)
diff2_rag = np.diff(seq_rag, n=2)

time_steps = np.arange(H)
time_steps_1 = np.arange(1, H)
time_steps_2 = np.arange(2, H)

# ==========================================
# 4. 开始画图
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

# --- 图1: 原始预测曲线对比 ---
axes[0].plot(time_steps, seq_true, label='真实曲线 (y)', color='black', linewidth=2.5, linestyle='--')
axes[0].plot(time_steps, seq_base, label='无RAG Base', color='red', linewidth=2, alpha=0.8, marker='o', markersize=4)
axes[0].plot(time_steps, seq_rag, label='加RAG (TS-RAG)', color='blue', linewidth=2, alpha=0.8, marker='s', markersize=4)
axes[0].set_title(f"1. 原始预测曲线对比 (Sample {SAMPLE_IDX}) - ", fontsize=14)
axes[0].set_ylabel("数值")
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# --- 图2: 一阶差分曲线 ---
axes[1].plot(time_steps_1, diff1_true, label='真实一阶差分', color='black', linewidth=2, linestyle='--')
axes[1].plot(time_steps_1, diff1_base, label='无RAG Base 一阶差分', color='red', linewidth=1.5, alpha=0.8)
axes[1].plot(time_steps_1, diff1_rag, label='加RAG 一阶差分', color='blue', linewidth=1.5, alpha=0.8)
axes[1].set_title("2. 一阶差分曲线 本身 ", fontsize=14)
axes[1].set_ylabel("Δy")
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

# --- 图3: 二阶差分曲线 ---
axes[2].plot(time_steps_2, diff2_true, label='真实二阶差分', color='black', linewidth=2, linestyle='--')
axes[2].plot(time_steps_2, diff2_base, label='无RAG Base 二阶差分', color='red', linewidth=1.5, alpha=0.8)
axes[2].plot(time_steps_2, diff2_rag, label='加RAG 二阶差分', color='blue', linewidth=1.5, alpha=0.8)
axes[2].set_title("3. 二阶差分曲线 本身 ", fontsize=14)
axes[2].set_xlabel("预测步长 t", fontsize=12)
axes[2].set_ylabel("Δ²y")
axes[2].legend(fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("my_results/Curves_Analysis.png", dpi=300)
print("✅ 三条曲线对比图已保存为 my_results/Curves_Analysis.png！")
plt.show()