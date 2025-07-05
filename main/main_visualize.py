# main_visualize.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
from src.utils import load_config

def main(args):
    config = load_config(args.config)
    file_path = os.path.join(config['output_dir'], args.file)
    
    if not os.path.exists(file_path):
        print(f"错误: 找不到数据文件 '{file_path}'。")
        return
        
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    if data.size == 0:
        print("错误: 文件中无有效数据。")
        return
        
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='deepskyblue', marker='.', s=5)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm) - Depth')
    ax.set_title('Visualization of Reconstructed 3D Profile')
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化重建的三维点云。")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径。')
    parser.add_argument('--file', type=str, default=None, help='要可视化的点云文件名 (位于输出目录中)。如果未提供，将使用配置文件中的默认值。')
    args = parser.parse_args()
    
    # 如果用户没有指定 --file 参数，则从配置文件中读取默认值
    if args.file is None:
        config = load_config(args.config)
        args.file = config['visualization']['point_cloud_file']
        
    main(args)