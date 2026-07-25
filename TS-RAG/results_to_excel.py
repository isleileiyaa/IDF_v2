import pandas as pd

# Baseline 评估结果数据
data = {
    '数据集': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'electricity', 'exchange_rate', 'weather'],
    'MSE': [0.3582, 0.2420, 0.3059, 0.1508, 0.1128, 0.0902, 0.1515],
    'MAE': [0.3650, 0.2986, 0.3206, 0.2283, 0.2008, 0.2084, 0.1872]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 计算平均值和最佳/最差表现
mean_mse = df['MSE'].mean()
mean_mae = df['MAE'].mean()
best_mse_dataset = df.loc[df['MSE'].idxmin(), '数据集']
best_mae_dataset = df.loc[df['MAE'].idxmin(), '数据集']
worst_mse_dataset = df.loc[df['MSE'].idxmax(), '数据集']
worst_mae_dataset = df.loc[df['MAE'].idxmax(), '数据集']

# 添加汇总行
summary_df = pd.DataFrame({
    '数据集': ['平均值', 'MSE最佳', 'MAE最佳', 'MSE最差', 'MAE最差'],
    'MSE': [mean_mse, df['MSE'].min(), '-', df['MSE'].max(), '-'],
    'MAE': [mean_mae, '-', df['MAE'].min(), '-', df['MAE'].max()]
})

# 合并数据
final_df = pd.concat([df, summary_df], ignore_index=True)

# 保存为 Excel 文件
output_path = 'results/forecast_evaluation/baseline_results.xlsx'
final_df.to_excel(output_path, index=False)

print(f"Excel 文件已保存到: {output_path}")
print("\n数据表格:")
print(final_df.to_string(index=False))
