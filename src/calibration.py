# src/calibration.py
import cv2
import numpy as np
import os
from typing import Tuple, Optional
from . import utils

class Calibrator:
    """
    封装相机和激光平面标定过程的类。
    """
    def __init__(self, config: dict):
        self.config = config
        self.cam_config = config['camera_calibration']
        self.laser_config = config['laser_calibration']
        self.data_root = config['data_root']
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)

    def calibrate_camera(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """执行相机标定。"""
        print("--> 开始相机标定...")
        pattern = tuple(self.cam_config['board_pattern'])
        square_size = self.cam_config['square_size']
        img_dir = os.path.join(self.data_root, self.cam_config['image_dir'])
        
        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size
        
        objpoints, imgpoints = [], []
        image_files = utils.get_image_files(img_dir)
        if not image_files: return None
        
        gray_shape = None
        for fname in image_files:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if gray_shape is None: gray_shape = img.shape[::-1]
            ret, corners = cv2.findChessboardCorners(img, pattern, None)
            if ret:
                objpoints.append(objp)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if not objpoints:
            print("错误: 没有足够的数据进行相机标定。")
            return None

        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
        if ret:
            print("\n相机标定成功!")
            output_path = os.path.join(self.output_dir, self.cam_config['output_file'])
            np.savez(output_path, mtx=mtx, dist=dist)
            print(f"相机参数已保存至 '{output_path}'")
            return mtx, dist
        else:
            print("\n相机标定失败!")
            return None

    def calibrate_laser_plane(self, mtx: np.ndarray, dist: np.ndarray) -> Optional[np.ndarray]:
        """执行激光平面标定。"""
        print("\n--> 开始激光平面标定...")
        pattern = tuple(self.cam_config['board_pattern'])
        square_size = self.cam_config['square_size']
        img_dir = os.path.join(self.data_root, self.laser_config['image_dir'])
        threshold = self.config['reconstruction']['laser_extraction_threshold']

        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_size
        
        all_3d_points = []
        image_files = utils.get_image_files(img_dir)
        if not image_files: return None

        for fname in image_files:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            ret, corners = cv2.findChessboardCorners(img, pattern, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
                
                laser_pixels = utils.extract_laser_line(img, threshold)
                if laser_pixels.size == 0: continue

                R, _ = cv2.Rodrigues(rvec)
                normal = R[:, 2]
                d_plane = np.dot(normal, tvec.flatten())
                
                undistorted_pixels = cv2.undistortPoints(laser_pixels, mtx, dist, P=mtx)
                inv_mtx = np.linalg.inv(mtx)
                
                for p in undistorted_pixels.reshape(-1, 2):
                    ray_dir = inv_mtx @ np.array([p[0], p[1], 1.0])
                    ray_dir /= np.linalg.norm(ray_dir)
                    dot_product = np.dot(normal, ray_dir)
                    if abs(dot_product) > 1e-6:
                        t = d_plane / dot_product
                        all_3d_points.append(t * ray_dir)
        
        if len(all_3d_points) < 100:
            print("错误: 收集到的3D点太少，无法拟合平面。")
            return None

        points_matrix = np.array(all_3d_points)
        centroid = np.mean(points_matrix, axis=0)
        U, _, Vt = np.linalg.svd(points_matrix - centroid)
        plane_normal = Vt[-1, :]
        D = -np.dot(plane_normal, centroid)
        plane_params = np.hstack((plane_normal, D))
        
        print("\n激光平面标定成功!")
        output_path = os.path.join(self.output_dir, self.laser_config['output_file'])
        np.savez(output_path, plane=plane_params)
        print(f"激光平面参数已保存至 '{output_path}'")
        return plane_params