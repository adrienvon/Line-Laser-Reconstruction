# src/utils.py
import cv2
import numpy as np
import os
import glob
import yaml
from typing import List, Tuple, Dict, Any

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """加载YAML配置文件。"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"配置文件 '{config_path}' 未找到。")
    except Exception as e:
        raise IOError(f"读取或解析配置文件时出错: {e}")

def get_image_files(dir_path: str) -> List[str]:
    """获取指定目录下所有支持的图像文件列表。"""
    supported_formats = ['*.bmp', '*.png', '*.jpg', '*.jpeg']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(dir_path, fmt)))
    if not image_files:
        print(f"警告: 在目录 '{dir_path}' 中未找到任何图像。")
    return image_files

def save_point_cloud(points: np.ndarray, file_path: str):
    """将三维点云保存为CSV文件。"""
    header = "X(mm),Y(mm),Z(mm)"
    np.savetxt(file_path, points, delimiter=",", header=header, comments="")
    print(f"点云数据已保存至 '{file_path}'")

def extract_laser_line(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    使用灰度重心法从图像中提取激光线中心（亚像素精度）。

    Args:
        image (np.ndarray): 输入的单通道灰度图。
        threshold (int): 用于过滤背景噪声的亮度阈值。

    Returns:
        np.ndarray: 激光线点的像素坐标数组，形状为 (N, 1, 2)。
    """
    height, width = image.shape
    laser_pixels = []
    
    for u in range(width):
        col = image[:, u].astype(np.float32)
        indices = np.where(col > threshold)[0]
        
        if len(indices) > 0:
            intensities = col[indices] - threshold
            weighted_sum = np.sum(indices * intensities)
            sum_of_intensities = np.sum(intensities)
            
            if sum_of_intensities > 1e-6:
                v_subpixel = weighted_sum / sum_of_intensities
                laser_pixels.append([u, v_subpixel])
            
    return np.array(laser_pixels, dtype=np.float32).reshape(-1, 1, 2)