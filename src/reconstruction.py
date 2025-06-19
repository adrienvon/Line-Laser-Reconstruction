# src/reconstruction.py
import cv2
import numpy as np
from . import utils

class Reconstructor:
    """
    封装三维重建过程的类。
    """
    def __init__(self, config: dict):
        self.config = config
        self.recon_config = config['reconstruction']
        self.threshold = self.recon_config['laser_extraction_threshold']

    def reconstruct_from_image(self, image: np.ndarray, mtx: np.ndarray, dist: np.ndarray, plane_params: np.ndarray) -> np.ndarray:
        """
        从单张图像重建三维轮廓。

        Args:
            image (np.ndarray): 输入的BGR图像。
            mtx (np.ndarray): 相机内参矩阵。
            dist (np.ndarray): 相机畸变系数。
            plane_params (np.ndarray): 激光平面方程 [A, B, C, D]。

        Returns:
            np.ndarray: N x 3 的三维点云数组。
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laser_pixels = utils.extract_laser_line(gray_image, self.threshold)
        
        if laser_pixels.size == 0:
            print("警告: 图像中未提取到激光线。")
            return np.array([])
            
        plane_normal = plane_params[:3]
        d_plane = plane_params[3]
        
        normalized_coords = cv2.undistortPoints(laser_pixels, mtx, dist, P=None)
        
        points_3d = []
        for nc in normalized_coords.reshape(-1, 2):
            ray_dir = np.array([nc[0], nc[1], 1.0])
            ray_dir /= np.linalg.norm(ray_dir)
            dot_product = np.dot(plane_normal, ray_dir)
            if abs(dot_product) > 1e-6:
                t = -d_plane / dot_product
                points_3d.append(t * ray_dir)
                
        return np.array(points_3d)