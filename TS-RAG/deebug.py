import numpy as np

# 加载数据 (替换为你自己的文件名)
y_true = np.load("my_results/ETTh1_retrieve_ETTh1_512_only_self_train_None_y_true.npy")
y_pred_rag = np.load("my_results/ETTh1_retrieve_ETTh1_512_only_self_train_None_y_pred_rag.npy")
y_pred_base = np.load("my_results/ETTh1_y_pred_base.npy")

# 展平
y_t = y_true.flatten()
y_r = y_pred_rag.flatten()
y_b = y_pred_base.flatten()

print("========== 数据极值与尺度诊断 ==========")
print(f"【真实值 y】      最大值: {np.max(y_t):.4f}, 最小值: {np.min(y_t):.4f}, 极差: {np.max(y_t)-np.min(y_t):.4f}")
print(f"【加RAG (rag)】   最大值: {np.max(y_r):.4f}, 最小值: {np.min(y_r):.4f}, 极差: {np.max(y_r)-np.min(y_r):.4f}")
print(f"【无RAG (base)】  最大值: {np.max(y_b):.4f}, 最小值: {np.min(y_b):.4f}, 极差: {np.max(y_b)-np.min(y_b):.4f}")
print("========================================")