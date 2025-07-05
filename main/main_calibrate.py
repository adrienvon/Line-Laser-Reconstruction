# main_calibrate.py
import os
import numpy as np
import argparse
from src.utils import load_config
from src.calibration import Calibrator

def main(args):
    config = load_config(args.config)
    calibrator = Calibrator(config)
    
    # 1. 相机标定
    cam_params = calibrator.calibrate_camera()
    if cam_params is None:
        print("相机标定失败，程序终止。")
        return
    mtx, dist = cam_params
    
    # 2. 激光平面标定
    plane_params = calibrator.calibrate_laser_plane(mtx, dist)
    if plane_params is None:
        print("激光平面标定失败，程序终止。")
        return
        
    print("\n所有标定流程成功完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机和激光平面标定程序。")
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径。')
    args = parser.parse_args()
    main(args)