import cv2
import numpy as np
import time
from joblib import load
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


# ======================== 定义 HogTransformer 类 ========================

class HogTransformer(BaseEstimator, TransformerMixin):
    """HOG特征提取器"""

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        from skimage.feature import hog
        hog_features = []
        for img in X:
            # 确保输入是二维图像
            if len(img.shape) == 3:
                img = img.squeeze()

            feat = hog(img,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       visualize=False)
            hog_features.append(feat)
        return np.array(hog_features)


# ======================== 模型加载 ========================

# 加载训练好的模型
MODEL_PATH = r'E:\code\Pythoncode\Pycharmproject\machinelearning\model\mnist_svm_model.joblib'
print(f"Loading model from {MODEL_PATH}...")
pipeline = load(MODEL_PATH)
print("Model loaded successfully!")


# ======================== 图像预处理函数 ========================

def prepare_digit_image(image):
    """
    预处理图像以匹配MNIST格式
    返回28x28灰度图像，黑底白字
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 增强对比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 自适应阈值二值化 - 反转方向匹配MNIST
    binary = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # 形态学操作（开运算）去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    try:
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        # 确保轮廓足够大
        if w < 10 or h < 10:
            return None

        # 扩展边界，确保数字完整
        margin = 15

        # 计算新边界
        new_x = max(0, x - margin)
        new_y = max(0, y - margin)
        new_w = min(cleaned.shape[1] - new_x, w + 2 * margin)
        new_h = min(cleaned.shape[0] - new_y, h + 2 * margin)

        # 检查尺寸有效性
        if new_w <= 0 or new_h <= 0:
            return None

        # 裁剪数字区域
        digit = cleaned[new_y:new_y + new_h, new_x:new_x + new_w]

        # 检查裁剪结果是否有效
        if digit.size == 0:
            return None

        # 调整大小为28x28
        resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

        # 转换为黑底白字 (匹配MNIST)
        inverted = cv2.bitwise_not(resized)

        return inverted

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


# ======================== 摄像头实时识别 ========================

def real_time_digit_recognition():
    """实时手写数字识别"""
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 获取摄像头分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建窗口
    cv2.namedWindow("Real-time Digit Recognition", cv2.WINDOW_NORMAL)

    # 处理间隔（毫秒）
    process_interval = 300  # 每300毫秒处理一次
    last_process_time = time.time()

    # 上一个预测结果
    last_prediction = "None"

    # 识别框位置和大小（居中，占屏幕高度的40%）
    box_width = int(height * 0.4)
    box_height = int(height * 0.4)
    box_x = (width - box_width) // 2
    box_y = (height - box_height) // 2

    # 背景图像（用于数字显示）
    digit_display = np.zeros((150, 150), dtype=np.uint8)

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # 创建显示图像的副本
        display_frame = frame.copy()

        # 绘制识别框
        cv2.rectangle(display_frame, (box_x, box_y),
                      (box_x + box_width, box_y + box_height),
                      (0, 255, 0), 2)

        # 提取ROI区域
        roi_frame = frame[box_y:box_y + box_height, box_x:box_x + box_width]

        # 检查是否到达处理间隔
        current_time = time.time()
        if (current_time - last_process_time) * 1000 >= process_interval:
            # 预处理ROI区域
            digit_img = prepare_digit_image(roi_frame)

            if digit_img is not None:
                try:
                    # 直接使用原始图像输入pipeline
                    # 确保输入形状为 (1, 28, 28)
                    input_img = digit_img.reshape(1, 28, 28)

                    # 进行预测
                    prediction = pipeline.predict(input_img)[0]
                    last_prediction = str(prediction)

                    # 更新数字显示
                    digit_display = cv2.resize(digit_img, (150, 150), interpolation=cv2.INTER_NEAREST)
                except Exception as e:
                    print(f"Prediction error: {e}")
                    last_prediction = "Error"
            else:
                last_prediction = "No digit"

            last_process_time = current_time

        # 在屏幕上显示预测结果
        cv2.putText(display_frame, f"Prediction: {last_prediction}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 显示说明文本
        cv2.putText(display_frame, "Write digit in green box",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示处理后的数字图像
        if digit_display is not None:
            # 转换为彩色以便在彩色图像上显示
            digit_color = cv2.cvtColor(digit_display, cv2.COLOR_GRAY2BGR)
            display_frame[20:170, width - 170:width - 20] = digit_color
            cv2.putText(display_frame, "Processed Digit",
                        (width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示帧
        cv2.imshow("Real-time Digit Recognition", display_frame)

        # 退出条件
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            # 按'c'键清除当前预测
            last_prediction = "None"
            digit_display = np.zeros((150, 150), dtype=np.uint8)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# ======================== 主程序 ========================

if __name__ == "__main__":
    real_time_digit_recognition()