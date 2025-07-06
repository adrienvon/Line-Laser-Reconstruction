import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def visualize_point_cloud(file_path):
    """
    读取CSV文件中的3D点云数据并进行可视化。
    :param file_path: 包含三维坐标的CSV文件路径。
    """
    # --- 1. 检查并加载数据 ---
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 '{file_path}'。")
        print("请先成功运行 'reconstruct_3d.py' 来生成该文件。")
        return

    print(f"正在从 '{file_path}' 读取数据...")
    try:
        # skiprows=1 是为了跳过CSV文件中的表头 (X(mm),Y(mm),Z(mm))
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 检查数据是否为空
    if data.size == 0 or data.ndim != 2 or data.shape[1] != 3:
        print("错误: 文件中没有有效的三维数据，无法进行可视化。")
        return
        
    print(f"成功加载 {data.shape[0]} 个三维点。")

    # --- 2. 创建3D图形 ---
    print("正在生成3D可视化图形...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # --- 3. 绘制散点图 ---
    # 使用data[:,0]作为X, data[:,1]作为Y, data[:,2]作为Z
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='deepskyblue', marker='.', s=5)

    # --- 4. 设置坐标轴和标题 ---
    # 使用更具描述性的标签，包含单位
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm) - Depth')
    ax.set_title('Visualization of Reconstructed 3D Profile')
    
    # 提示：根据你的相机和物体摆放，可能需要反转某个轴才能获得更直观的视图
    # 例如，如果深度方向看起来是反的，可以取消下面这行的注释
    # ax.invert_zaxis()

    # 设置坐标轴比例，使视图更真实
    # 注意: 'equal' 可能会在某些matplotlib版本中导致拉伸，'auto'是默认值
    ax.set_aspect('auto') 

    # --- 5. 显示图形 ---
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 定义要可视化的数据文件
    INPUT_FILE = "./cal01/add02.csv"
    
    # 调用可视化函数
    visualize_point_cloud(INPUT_FILE)