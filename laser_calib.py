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

    images = glob.glob(os.path.join(image_dir, '*.bmp'))
    if not images:
        print(f"错误: 在 '{image_dir}' 中没有找到图像。请检查路径。")
        return None, None, None
        
    print(f"找到 {len(images)} 张相机标定图像。")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, board_pattern, None)

        # 如果找到，添加对象点和图像点 (经过亚像素精确化后)
        if ret == True:
            objpoints.append(objp)
            # 亚像素角点优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"成功找到角点: {os.path.basename(fname)}")
        else:
            print(f"警告: 未能找到角点: {os.path.basename(fname)}")

    if not objpoints or not imgpoints:
        print("错误: 没有足够的数据进行标定。")
        return None, None, None

    # 进行相机标定
    print("\n正在计算相机参数...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("\n相机标定成功!")
        print("相机内参矩阵 K:\n", mtx)
        print("畸变系数 D:\n", dist)
        
        # 保存标定结果
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

def extract_laser_line_pixels(image):
    """
    从图像中提取激光线的中心像素坐标。
    这是一个简化的实现，实际应用中可能需要更复杂的算法（如Steger）。
    这里我们使用一个简单的方法：在每一列找到最亮的点。
    :param image: 输入的单通道灰度图。
    :return: 激光线点的像素坐标列表 [(u, v), ...]。
    """
    height, width = image.shape
    laser_pixels = []
    
    # 设定一个亮度阈值，过滤掉噪声
    BRIGHTNESS_THRESHOLD = 150

    for u in range(width):
        col = image[:, u]
        max_val = np.max(col)
        if max_val > BRIGHTNESS_THRESHOLD:
            v = np.argmax(col)
            laser_pixels.append([u, v])
            
    return np.array(laser_pixels, dtype=np.float32).reshape(-1, 1, 2)


def calibrate_laser_plane(image_dir, board_pattern, square_size, mtx, dist):
    """
    标定激光平面。
    :param image_dir: 存放激光标定图像的文件夹路径。
    :param board_pattern: 棋盘格内部角点数量。
    :param square_size: 棋盘格方格尺寸。
    :param mtx: 相机内参矩阵。
    :param dist: 相机畸变系数。
    :return: 激光平面方程的系数 (A, B, C, D)，标定结果的路径。
    """
    print("\n--> 开始激光平面标定...")
    
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    all_laser_points_3d = []  # 用于存储所有激光点在相机坐标系下的3D坐标

    images = glob.glob(os.path.join(image_dir, '*.bmp'))
    if not images:
        print(f"错误: 在 '{image_dir}' 中没有找到图像。请检查路径。")
        return None, None
        
    print(f"找到 {len(images)} 张激光标定图像。")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 找到棋盘格位姿
        ret, corners = cv2.findChessboardCorners(gray, board_pattern, None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 使用solvePnP计算标定板的旋转和平移
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
            print(f"处理图像: {os.path.basename(fname)}")

            # 2. 提取激光线像素点
            laser_pixels = extract_laser_line_pixels(gray)
            if laser_pixels.size == 0:
                print(f"警告: 在 {os.path.basename(fname)} 中未提取到激光线。")
                continue

            # 3. 将2D激光像素点转换为相机坐标系下的3D点
            # 这一步是核心：求解从相机光心发出的射线与标定板平面的交点

            # 首先，将旋转向量转换为旋转矩阵
            R, _ = cv2.Rodrigues(rvec)
            
            # 标定板平面的法向量（在相机坐标系中）是旋转矩阵R的第三列
            normal = R[:, 2]
            
            # 标定板平面上的一个点（在相机坐标系中）就是平移向量tvec
            # 平面方程为: normal · (X - tvec) = 0  =>  normal · X = normal · tvec
            d = np.dot(normal, tvec.flatten())

            # 反投影激光像素点，得到相机坐标系中的方向向量
            # 先去畸变
            undistorted_laser_pixels = cv2.undistortPoints(laser_pixels, mtx, dist, P=mtx)
            
            for p in undistorted_laser_pixels.reshape(-1, 2):
                # 构造射线方向向量
                ray_dir = np.array([p[0], p[1], 1.0])
                ray_dir = np.linalg.inv(mtx) @ ray_dir
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # 求解射线与平面的交点
                # 射线方程: P = t * ray_dir  (相机原点为(0,0,0))
                # 平面方程: normal · P = d
                # 代入得: t * (normal · ray_dir) = d => t = d / (normal · ray_dir)
                t = d / np.dot(normal, ray_dir)
                point_3d = t * ray_dir
                all_laser_points_3d.append(point_3d)
        else:
            print(f"警告: 在 {os.path.basename(fname)} 中未找到角点，跳过。")

    if len(all_laser_points_3d) < 100: # 至少需要一些点来拟合
        print("错误: 收集到的3D激光点太少，无法进行平面拟合。")
        return None, None
        
    print(f"\n收集到 {len(all_laser_points_3d)} 个3D激光点，开始拟合平面...")

    # 4. 使用SVD(奇异值分解)拟合平面
    points_matrix = np.array(all_laser_points_3d)
    centroid = np.mean(points_matrix, axis=0) # 计算点云的质心
    centered_points = points_matrix - centroid # 中心化点云
    
    U, S, Vt = np.linalg.svd(centered_points)
    
    # 平面的法向量是对应最小奇异值的右奇异向量 (Vt的最后一行)
    plane_normal = Vt[-1, :]
    A, B, C = plane_normal
    
    # D = -(A*xc + B*yc + C*zc)
    D = -np.dot(plane_normal, centroid)
    
    plane_params = np.array([A, B, C, D])
    
    print("\n激光平面标定成功!")
    print(f"平面方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
    
    # 保存结果
    output_path = "laser_plane_calibration.npz"
    np.savez(output_path, plane=plane_params)
    print(f"激光平面参数已保存至 '{output_path}'")
    
    return plane_params, output_path

# ------------------------------------------------------------------------------------
# 第3部分: 主程序
# ------------------------------------------------------------------------------------
def main():
    # --- 参数配置 ---
    # 棋盘格内部角点数量 (corners_col, corners_row)
    BOARD_PATTERN = (4, 7)
    # 棋盘格方块的实际大小 (单位: mm)
    SQUARE_SIZE = 10.0
    # 图像文件夹路径
    CAMERA_CALIB_DIR = 'mvs/camera'
    LASER_CALIB_DIR = 'mvs/line'
    camera_calib_file = CAMERA_CALIB_DIR
    # --- 步骤1: 相机标定 ---
    mtx, dist, camera_calib_file = calibrate_camera(CAMERA_CALIB_DIR, BOARD_PATTERN, SQUARE_SIZE)
    
    if mtx is None:
        print("\n相机标定失败，程序终止。")
        return

    # --- 步骤2: 激光平面标定 ---
    # 如果相机参数已存在，也可以选择直接加载
    if os.path.exists("camera_calibration.npz"):
        print("\n加载已有的相机标定文件...")
        with np.load("camera_calibration.npz") as data:
            mtx = data['mtx']
            dist = data['dist']   

    plane_params, laser_calib_file = calibrate_laser_plane(LASER_CALIB_DIR, BOARD_PATTERN, SQUARE_SIZE, mtx, dist)

    if plane_params is None:
        print("\n激光平面标定失败，程序终止。")
        return
        
    print("\n\n所有标定流程完成！")
    print(f"相机参数保存在: {camera_calib_file}")
    print(f"激光平面参数保存在: {laser_calib_file}")


if __name__ == '__main__':
    main()