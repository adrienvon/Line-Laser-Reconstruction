import pandas as pd
from sklearn.cluster import DBSCAN

def analyze_csv_gradients(file_path, column_index, eps=2.5, min_samples=5):
    """
    分析CSV文件中指定列的数据梯度。

    该函数会自动处理带标题行的CSV文件，对指定列进行DBSCAN聚类以识别
    稳定的数据平台（梯度），然后计算每个平台的统计数据及其平均值之间的差值。

    参数:
    file_path (str): CSV文件的路径。
    column_index (int): 需要分析的列的索引（0代表第一列，1代表第二列，以此类推）。
    eps (float): DBSCAN算法的'eps'参数。
    min_samples (int): DBSCAN算法的'min_samples'参数。
    """
    try:
        # 1. 加载数据
        # 优化：不再使用 header=None，pandas会自动将第一行作为标题。
        df = pd.read_csv(file_path)
        
        if column_index >= len(df.columns):
            print(f"错误: 列索引 {column_index} 超出范围。文件只有 {len(df.columns)} 列。")
            return
        
        # 2. 提取并清洗数据
        # 按整数位置选择列
        y_values_series = df.iloc[:, column_index]
        
        # 优化：强制转换为数字，任何无法转换的值（如文本）都会变成NaN（空值）
        numeric_values = pd.to_numeric(y_values_series, errors='coerce')
        
        # 去除所有空值（包括原有的和转换失败的）
        cleaned_values = numeric_values.dropna()
        
        if cleaned_values.empty:
            print(f"错误: 第 {column_index + 1} 列在移除标题和非数字值后为空。请检查文件内容和指定的列号。")
            return

        print(f"--- 正在分析文件: '{file_path}', 第 {column_index + 1} 列 (已跳过标题行) ---")

        # 3. DBSCAN 聚类
        data_for_clustering = cleaned_values.values.reshape(-1, 1)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(data_for_clustering)
        
        # 建立一个新的DataFrame用于分析，确保数据和聚类标签对齐
        analysis_df = pd.DataFrame({'Value': cleaned_values.values, 'Cluster': clusters})
        
        # 4. 计算每个梯度的平均值 (排除异常值, Cluster == -1)
        valid_clusters = analysis_df[analysis_df['Cluster'] != -1]
        
        if valid_clusters.empty:
            num_outliers = len(analysis_df[analysis_df['Cluster'] == -1])
            print(f"\n分析完成：未能识别出任何稳定的梯度平台。共 {num_outliers} 个点被识别为异常值。")
            print("建议尝试调整 'eps' 参数（例如，增大该值）来放宽聚类条件。")
            return

        # 5. 计算统计数据
        print("\n每个梯度的统计情况：")
        gradient_stats = valid_clusters.groupby('Cluster')['Value'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)
        gradient_stats.rename(columns={'mean': '平均值', 'std': '标准差', 'count': '点数'}, inplace=True)
        print(gradient_stats.to_string(formatters={'平均值': '{:.2f}'.format, '标准差': '{:.2f}'.format}))
        
        # 6. 计算差值
        gradient_means = gradient_stats['平均值']
        if len(gradient_means) > 1:
            mean_values = gradient_means.tolist()
            mean_indices = gradient_means.index.tolist()
            
            print("\n已排序的梯度平均值之间的差值：")
            for i in range(len(mean_values) - 1):
                diff = mean_values[i] - mean_values[i+1]
                id1, id2 = mean_indices[i], mean_indices[i+1]
                print(f"  > 梯度 {id1} ({mean_values[i]:.2f}) 与 梯度 {id2} ({mean_values[i+1]:.2f}) 的差值: {diff:.2f}")
        else:
            print("\n只识别出一个梯度，无法计算差值。")
            
        num_outliers = len(analysis_df[analysis_df['Cluster'] == -1])
        print(f"\n分析完成：共识别出 {len(gradient_means)} 个梯度平台和 {num_outliers} 个异常（过渡）点。")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查文件路径是否正确。")
    except Exception as e:
        print(f"处理过程中发生未知错误: {e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 请在这里修改参数 ---
    
    # 1. 指定要分析的CSV文件路径
    # 示例: file_to_analyze = "cal01/add01.csv"
    file_to_analyze = "cal01/add01.csv"  # <--- 修改这里
    
    # 2. 指定要分析的列（0是第一列，1是第二列）
    column_to_analyze = 1 # <--- 修改这里，分析第二列
    
    # 3. (可选) 调整DBSCAN参数
    epsilon_value = 0.8
    min_points = 5
    
    # --- 执行分析 ---
    analyze_csv_gradients(
        file_path=file_to_analyze, 
        column_index=column_to_analyze,
        eps=epsilon_value,
        min_samples=min_points
    )