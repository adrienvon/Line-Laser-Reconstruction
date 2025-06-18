import cv2
import numpy as np
import os

def extract_laser_line_pixels_subpixel(image):
    """
    从图像中提取激光线的中心像素坐标 (使用灰度重心法，达到亚像素精度)。
    这个函数必须与标定脚本中的版本完全一致，以保证算法的一致性。
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
            # 减去阈值可以进一步减小背景噪声的影响
            weighted_sum = np.sum(bright_pixels_indices * (intensities - BRIGHTNESS_THRESHOLD))
            sum_of_intensities = np.sum(intensities - BRIGHTNESS_THRESHOLD)
            
            if sum_of_intensities > 0:
                v_subpixel = weighted_sum / sum_of_intensities
                laser_pixels.append([u, v_subpixel])
            
    return np.array(laser_pixels, dtype=np.float32).reshape(-1, 1, 2)


def get_3d_points_from_image(image, mtx, dist, plane_params):
    """
    从单张带有激光线的图像中计算三维点云轮廓。
    :param image: 输入的BGR图像。
    :param mtx: 相机内参矩阵。
    :param dist: 相机畸变系数。
    :param plane_params: 激光平面方程 [A, B, C, D]。
    :return: N x 3 的 numpy 数组，包含轮廓线上所有点的3D坐标。
    """
    # 1. 提取激光线像素点 (使用与标定相同的亚像素方法)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laser_pixels = extract_laser_line_pixels_subpixel(gray)
    
    if laser_pixels.size == 0:
        print("警告: 图像中未提取到激光线。")
        return np.array([])

    # 2. 准备平面和相机参数
    A, B, C, D = plane_params
    plane_normal = np.array([A, B, C])
    inv_mtx = np.linalg.inv(mtx)

    # 3. 对像素点进行去畸变，并一次性转换为归一化坐标
    # 使用 cv2.undistortPoints 可以直接获得去畸变后的归一化坐标 (x, y)
    # 这样就不需要再乘以内参的逆了
    normalized_coords = cv2.undistortPoints(laser_pixels, mtx, dist, P=None)

    points_3d = []
    # 4. 循环计算每个点的3D坐标
    for nc in normalized_coords.reshape(-1, 2):
        # 构造射线方向向量
        ray_dir = np.array([nc[0], nc[1], 1.0]) # 归一化坐标系下的方向
        ray_dir /= np.linalg.norm(ray_dir) # 标准化为单位向量

        # 求解射线与平面的交点
        # t = -D / (N · V)
        dot_product = np.dot(plane_normal, ray_dir)
        if abs(dot_product) > 1e-6: # 避免除以零
            t = -D / dot_product
            point_3d = t * ray_dir
            points_3d.append(point_3d)

    return np.array(points_3d)

def main():
    # --- 1. 定义标定文件的路径 ---
    CAMERA_CALIB_FILE = "camera_calibration.npz"
    LASER_CALIB_FILE = "laser_plane_calibration.npz"

    # --- 2. 加载标定参数 ---
    print("正在加载标定参数...")
    try:
        with np.load(CAMERA_CALIB_FILE) as data:
            mtx = data['mtx']
            dist = data['dist']
        with np.load(LASER_CALIB_FILE) as data:
            plane_params = data['plane']
    except FileNotFoundError as e:
        print(f"错误: 找不到标定文件 '{e.filename}'。")
        print("请确保已成功运行标定脚本 'calibrate.py' 并且 .npz 文件在当前目录。")
        return

    print("相机内参 K:\n", mtx)
    print("畸变系数 D:\n", dist)
    print(f"激光平面方程: {plane_params[0]:.6f}x + {plane_params[1]:.6f}y + {plane_params[2]:.6f}z + {plane_params[3]:.6f} = 0")
    print("标定参数加载成功！")

    # --- 3. 指定要重建的图像 ---
    # 请将 'test_object.png' 替换为您自己的图像文件名
    test_image_path = 'mvs/st/Image_15189.bmp' 
    
    if not os.path.exists(test_image_path):
        print(f"\n错误: 找不到测试图像 '{test_image_path}'。")
        print("请在项目文件夹中放置一张名为 'test_object.png' 的图像，或修改脚本中的文件名。")
        return
    
    image_to_measure = cv2.imread(test_image_path)
    if image_to_measure is None:
        print(f"错误: 无法读取图像 '{test_image_path}'。请检查文件是否损坏。")
        return
        
    # --- 4. 执行三维重建 ---
    print(f"\n正在从 '{test_image_path}' 重建三维轮廓...")
    object_profile_3d = get_3d_points_from_image(image_to_measure, mtx, dist, plane_params)

    if object_profile_3d.size > 0:
        print(f"成功重建 {len(object_profile_3d)} 个三维点。")
        
        # --- 5. 保存并显示结果 ---
        output_file = "reconstructed_profile.csv"
        # 保存为CSV格式，X,Y,Z三列，单位为毫米
        np.savetxt(output_file, object_profile_3d, delimiter=",", header="X(mm),Y(mm),Z(mm)", comments="")
        print(f"三维轮廓数据已保存到 '{output_file}'")
        
        # 打印前5个点作为示例
        print("\n前5个点的三维坐标 (X, Y, Z) in mm:")
        # 设置打印选项，使其更易读
        np.set_printoptions(precision=3, suppress=True)
        print(object_profile_3d[:5])
    else:
        print("重建失败：未能从图像中获取任何三维点。")


if __name__ == '__main__':
    main()