import cv2
import numpy as np
import os
import glob

# ------------------------------------------------------------------------------------
# 第1部分: 相机标定
# ------------------------------------------------------------------------------------
def calibrate_camera(image_dir, board_pattern, square_size):
    """
    对相机进行标定。
    :param image_dir: 存放相机标定图像的文件夹路径。
    :param board_pattern: 棋盘格内部角点的数量, (列数, 行数)，例如 (9, 6)。
    :param square_size: 棋盘格每个方格的实际尺寸（毫米mm）。
    :return: 相机内参矩阵, 畸变系数, 标定结果的路径。
    """
    print("--> 开始相机标定...")

    # 准备对象点, 如 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp = objp * square_size

    # 用于存储所有图像的对象点和图像点的列表
    objpoints = []  # 3d点在世界坐标系中的位置
    imgpoints = []  # 2d点在图像平面中的位置

    # 支持多种图像格式
    images = []
    for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
        images.extend(glob.glob(os.path.join(image_dir, ext)))

    if not images:
        print(f"错误: 在 '{image_dir}' 中没有找到图像。请检查路径。")
        return None, None, None
        
    print(f"找到 {len(images)} 张相机标定图像。")
    gray_shape = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1]

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, board_pattern, None)

        if ret == True:
            objpoints.append(objp)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"成功找到角点: {os.path.basename(fname)}")
        else:
            print(f"警告: 未能找到角点: {os.path.basename(fname)}")

    if not objpoints:
        print("错误: 没有足够的数据进行标定。")
        return None, None, None

    print("\n正在计算相机参数...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    if ret:
        print("\n相机标定成功!")
        print("相机内参矩阵 K:\n", mtx)
        print("畸变系数 D:\n", dist)
        
        output_path = "camera_calibration.npz"
        np.savez(output_path, mtx=mtx, dist=dist)
        print(f"标定结果已保存至 '{output_path}'")
        return mtx, dist, output_path
    else:
        print("\n相机标定失败!")
        return None, None, None

# ------------------------------------------------------------------------------------
# 第2部分: 激光平面标定
# ------------------------------------------------------------------------------------

def extract_laser_line_pixels_subpixel(image):
    """
    从图像中提取激光线的中心像素坐标 (使用灰度重心法，达到亚像素精度)。
    :param image: 输入的单通道灰度图。
    :return: 激光线点的像素坐标列表 [(u, v), ...]。
    """
    height, width = image.shape
    laser_pixels = []
    
    # 亮度阈值，过滤掉大部分背景噪声
    BRIGHTNESS_THRESHOLD = 50 

    for u in range(width):
        col = image[:, u].astype(np.float32)
        bright_pixels_indices = np.where(col > BRIGHTNESS_THRESHOLD)[0]
        
        if len(bright_pixels_indices) > 0:
            intensities = col[bright_pixels_indices]
            weighted_sum = np.sum(bright_pixels_indices * (intensities - BRIGHTNESS_THRESHOLD))
            sum_of_intensities = np.sum(intensities - BRIGHTNESS_THRESHOLD)
            
            if sum_of_intensities > 0:
                v_subpixel = weighted_sum / sum_of_intensities
                laser_pixels.append([u, v_subpixel])
            
    return np.array(laser_pixels, dtype=np.float32).reshape(-1, 1, 2)


def calibrate_laser_plane(image_dir, board_pattern, square_size, mtx, dist):
    """
    标定激光平面。
    """
    print("\n--> 开始激光平面标定...")
    
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    all_laser_points_3d = []

    images = []
    for ext in ['*.bmp', '*.png', '*.jpg', '*.jpeg']:
        images.extend(glob.glob(os.path.join(image_dir, ext)))

    if not images:
        print(f"错误: 在 '{image_dir}' 中没有找到图像。")
        return None, None
        
    print(f"找到 {len(images)} 张激光标定图像。")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, board_pattern, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            print(f"处理图像: {os.path.basename(fname)}")

            laser_pixels = extract_laser_line_pixels_subpixel(gray)
            if laser_pixels.size == 0:
                print(f"警告: 在 {os.path.basename(fname)} 中未提取到激光线。")
                continue

            R, _ = cv2.Rodrigues(rvec)
            normal = R[:, 2]
            d = np.dot(normal, tvec.flatten())
            undistorted_laser_pixels = cv2.undistortPoints(laser_pixels, mtx, dist, P=mtx)
            
            for p in undistorted_laser_pixels.reshape(-1, 2):
                ray_dir = np.linalg.inv(mtx) @ np.array([p[0], p[1], 1.0])
                ray_dir /= np.linalg.norm(ray_dir)

                # 避免除以零的边缘情况
                dot_product = np.dot(normal, ray_dir)
                if abs(dot_product) > 1e-6:
                    t = d / dot_product
                    point_3d = t * ray_dir
                    all_laser_points_3d.append(point_3d)
        else:
            print(f"警告: 在 {os.path.basename(fname)} 中未找到角点，跳过。")

    if len(all_laser_points_3d) < 100:
        print("错误: 收集到的3D激光点太少，无法进行平面拟合。")
        return None, None
        
    print(f"\n收集到 {len(all_laser_points_3d)} 个3D激光点，开始拟合平面...")

    points_matrix = np.array(all_laser_points_3d)
    centroid = np.mean(points_matrix, axis=0)
    centered_points = points_matrix - centroid
    
    U, S, Vt = np.linalg.svd(centered_points)
    
    plane_normal = Vt[-1, :]
    A, B, C = plane_normal
    D = -np.dot(plane_normal, centroid)
    
    plane_params = np.array([A, B, C, D])
    
    print("\n激光平面标定成功!")
    # 增加输出精度，便于观察
    print(f"平面方程: {A:.6f}x + {B:.6f}y + {C:.6f}z + {D:.6f} = 0")
    
    output_path = "laser_plane_calibration.npz"
    np.savez(output_path, plane=plane_params)
    print(f"激光平面参数已保存至 '{output_path}'")
    
    return plane_params, output_path

# ------------------------------------------------------------------------------------
# 第3部分: 主程序
# ------------------------------------------------------------------------------------
def main():
    # --- 参数配置 ---
    BOARD_PATTERN = (4, 7)
    SQUARE_SIZE = 10.0
    CAMERA_CALIB_DIR = 'mvs/camera'
    LASER_CALIB_DIR = 'mvs/line'
    CAMERA_CALIB_FILE = "camera_calibration.npz"

    # --- 步骤1: 获取相机参数 ---
    # 逻辑：如果标定文件存在，直接加载；否则，执行标定。
    if os.path.exists(CAMERA_CALIB_FILE):
        print(f"加载已有的相机标定文件: {CAMERA_CALIB_FILE}")
        with np.load(CAMERA_CALIB_FILE) as data:
            mtx = data['mtx']
            dist = data['dist']
    else:
        print(f"未找到相机标定文件 '{CAMERA_CALIB_FILE}'，将从图像进行标定...")
        mtx, dist, _ = calibrate_camera(CAMERA_CALIB_DIR, BOARD_PATTERN, SQUARE_SIZE)
    
    if mtx is None:
        print("\n相机参数获取失败，程序终止。")
        return

    # --- 步骤2: 执行激光平面标定 ---
    # 这里我们总是执行激光平面标定，因为通常这一步需要根据最新的相机参数进行
    plane_params, laser_calib_file = calibrate_laser_plane(LASER_CALIB_DIR, BOARD_PATTERN, SQUARE_SIZE, mtx, dist)

    if plane_params is None:
        print("\n激光平面标定失败，程序终止。")
        return
        
    print("\n\n所有标定流程完成！")
    print(f"相机参数从 '{CAMERA_CALIB_FILE}' 加载或生成。")
    print(f"激光平面参数已保存在: {laser_calib_file}")

if __name__ == '__main__':
    main()