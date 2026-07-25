import numpy as np
import matplotlib.pyplot as plt
import os

print("正在加载数据，请稍候...")

# 1. 对应你刚才跑出来的真实文件名
file_rag_prefix = "my_results/ETTh1_retrieve_ETTh1_512_only_self_train_None"
file_base = "my_results/ETTh1_y_pred_base.npy"

try:
    # 加载真实值、RAG预测值、Base预测值
    y_true = np.load(f"{file_rag_prefix}_y_true.npy")
    y_pred_rag = np.load(f"{file_rag_prefix}_y_pred_rag.npy")
    y_pred_base = np.load(file_base)
    print("✅ 三个数据文件全部加载成功！")
except FileNotFoundError as e:
    print(f"❌ 找不到文件，请检查 my_results 文件夹里的文件名是否一致: {e}")
    exit()

# 2. 数据形状处理 (去掉多余的 batch 维度: (1, 19719, 64) -> (19719, 64))
y_true = np.squeeze(y_true, axis=0)
y_pred_rag = np.squeeze(y_pred_rag, axis=0)
y_pred_base = np.squeeze(y_pred_base, axis=0)

# 取最后一个特征列（目标列OT）来画图
if len(y_true.shape) == 3:
    y_true = y_true[:, :, -1]
    y_pred_rag = y_pred_rag[:, :, -1]
    y_pred_base = y_pred_base[:, :, -1]

# 为了画图清晰，我们取前 500 个测试时间点（画两万个点全糊在一起了）
SAMPLE_NUM = 500
y_true = y_true[:SAMPLE_NUM, :]
y_pred_rag = y_pred_rag[:SAMPLE_NUM, :]
y_pred_base = y_pred_base[:SAMPLE_NUM, :]

# 展平数据用于计算全局统计量
y_t_flat = y_true.flatten()
y_b_flat = y_pred_base.flatten()
y_r_flat = y_pred_rag.flatten()

H = y_true.shape[1] # 预测步长 (64)

# ==========================================
# 3. 开始画图
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(16, 10))

# --- 指标1: 均值与方差 ---
print("\n==========  1：均值与方差 ==========")
print(f"真实值 y    - 均值: {np.mean(y_t_flat):.4f}, 方差: {np.var(y_t_flat):.4f}")
print(f"Base无RAG   - 均值: {np.mean(y_b_flat):.4f}, 方差: {np.var(y_b_flat):.4f}")
print(f"加了 TS-RAG - 均值: {np.mean(y_r_flat):.4f}, 方差: {np.var(y_r_flat):.4f}")
print("==================================================\n")

# --- 图2: 绝对误差分布直方图 ---
ax1 = plt.subplot(2, 2, 1)
ax1.hist(np.abs(y_b_flat - y_t_flat), bins=50, alpha=0.5, label='无RAG Base', color='red', density=True)
ax1.hist(np.abs(y_r_flat - y_t_flat), bins=50, alpha=0.5, label='加了 TS-RAG', color='blue', density=True)
ax1.set_title("指标2. 绝对误差分布 |y_hat - y|")
ax1.set_xlabel("绝对误差大小")
ax1.set_ylabel("频率")
ax1.legend()

# --- 图3: 期望偏差 E[y_hat - y] ---
ax2 = plt.subplot(2, 2, 2)
bias_base = np.mean(y_b_flat - y_t_flat)
bias_rag = np.mean(y_r_flat - y_t_flat)
ax2.bar(['无RAG (Base)', '加RAG (TS-RAG)'], [bias_base, bias_rag], color=['red', 'blue'], alpha=0.7, width=0.4)
ax2.axhline(0, color='black', linestyle='--')
ax2.set_title("指标3. 误差期望 E[y_hat - y] (系统性偏差)")
ax2.set_ylabel("偏差值 Bias")
for i, v in enumerate([bias_base, bias_rag]):
    ax2.text(i, v, f"{v:.4f}", ha='center', va='bottom' if v>0 else 'top', fontsize=12)

# --- 图4: 一阶/二阶差分误差折线图 ---
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

diff1_y = np.diff(y_true, n=1, axis=1)
diff1_b = np.diff(y_pred_base, n=1, axis=1)
diff1_r = np.diff(y_pred_rag, n=1, axis=1)

diff2_y = np.diff(y_true, n=2, axis=1)
diff2_b = np.diff(y_pred_base, n=2, axis=1)
diff2_r = np.diff(y_pred_rag, n=2, axis=1)

err_diff1_b = np.mean(np.abs(diff1_b - diff1_y), axis=0)
err_diff1_r = np.mean(np.abs(diff1_r - diff1_y), axis=0)
err_diff2_b = np.mean(np.abs(diff2_b - diff2_y), axis=0)
err_diff2_r = np.mean(np.abs(diff2_r - diff2_y), axis=0)

time_steps_1 = np.arange(1, H)
time_steps_2 = np.arange(2, H)

ax3.plot(time_steps_1, err_diff1_b, label='无RAG', color='red', marker='o', markersize=3)
ax3.plot(time_steps_1, err_diff1_r, label='加RAG', color='blue', marker='x', markersize=3)
ax3.set_title("指标4a. 一阶差分绝对误差 |Δy_hat - Δy| (斜率误差)")
ax3.set_xlabel("预测步长 t")
ax3.legend()

ax4.plot(time_steps_2, err_diff2_b, label='无RAG', color='red', marker='o', markersize=3)
ax4.plot(time_steps_2, err_diff2_r, label='加RAG', color='blue', marker='x', markersize=3)
ax4.set_title("指标4b. 二阶差分绝对误差 |Δ²y_hat - Δ²y| (拐点/细节误差)")
ax4.set_xlabel("预测步长 t")
ax4.legend()

plt.tight_layout()
plt.savefig("my_results/Final_Report_Plots.png", dpi=300)
print("✅ 图表已保存为 my_results/Final_Report_Plots.png！即将弹出展示...")
plt.show()