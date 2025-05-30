import os
import numpy as np
import struct
import cv2
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
from joblib import dump, load
import pickle
import matplotlib.pyplot as plt
import time


# ======================== MNIST二进制数据加载模块 ========================

class MNISTLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_images(self, filename):
        """加载MNIST图像二进制文件"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_labels(self, filename):
        """加载MNIST标签二进制文件"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    def load_dataset(self):
        """加载完整训练集和测试集"""
        X_train = self.load_images('train-images-idx3-ubyte')
        y_train = self.load_labels('train-labels-idx1-ubyte')
        X_test = self.load_images('t10k-images-idx3-ubyte')
        y_test = self.load_labels('t10k-labels-idx1-ubyte')
        return X_train, y_train, X_test, y_test


# ======================== 特征工程模块 ========================

class HogTransformer(BaseEstimator, TransformerMixin):
    """HOG特征提取器（修复维度问题）"""

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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


# ======================== 模型训练与评估模块 ========================

class SVMTrainer:
    def __init__(self, use_hog=True, use_stacking=True):
        self.use_hog = use_hog
        self.use_stacking = use_stacking
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """构建预处理和模型管道"""
        steps = []

        # 1. 特征工程选择
        if self.use_hog:
            steps.append(('hog', HogTransformer()))
        else:
            # 如果是原始像素，需要展平
            steps.append(('flatten', lambda X: X.reshape(X.shape[0], -1)))

        # 2. 数据标准化
        steps.append(('scaler', MinMaxScaler()))

        # 3. 模型选择
        if self.use_stacking:
            base_models = [
                ('svm1', svm.SVC(kernel='rbf', C=10, gamma=0.01, probability=True)),
                ('svm2', svm.SVC(kernel='rbf', C=50, gamma=0.005, probability=True))
            ]
            final_model = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(max_iter=1000),
                stack_method='predict_proba'
            )
            steps.append(('model', final_model))
        else:
            steps.append(('model', svm.SVC(kernel='rbf')))

        return Pipeline(steps)

    def train(self, X_train, y_train):
        """训练模型（自动网格搜索）"""
        param_grid = {}

        # 网格搜索参数配置
        if not self.use_stacking:
            param_grid.update({
                'model__C': [1, 10, 100],
                'model__gamma': [0.001, 0.01, 0.1],
                'model__class_weight': [None, 'balanced']
            })

        # 执行网格搜索
        if param_grid:
            self.pipeline = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )

        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        print(f"训练时间: {time.time() - start_time:.2f}秒")

        if hasattr(self.pipeline, 'best_params_'):
            print("最佳参数:", self.pipeline.best_params_)

    def evaluate(self, X_test, y_test, show_plot=False):
        """评估模型性能
        Args:
            show_plot: 是否显示混淆矩阵图像
        """
        y_pred = self.pipeline.predict(X_test)

        # 输出完整分类报告
        print("\n分类报告:")
        print(metrics.classification_report(y_test, y_pred, digits=4))

        # 创建保存图像的目录
        img_dir = r'E:\code\Pythoncode\Pycharmproject\machinelearning\data\img'
        os.makedirs(img_dir, exist_ok=True)

        # 混淆矩阵可视化
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(img_dir, f'confusion_matrix_{timestamp}.png')

        disp = metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            cmap=plt.cm.Blues,
            normalize='true'
        )
        plt.title("标准化混淆矩阵")

        # 保存图像
        plt.savefig(img_path)
        print(f"混淆矩阵已保存至: {img_path}")

        # 根据参数决定是否显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()

        # 计算各项指标
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='macro')
        recall = metrics.recall_score(y_test, y_pred, average='macro')
        f1 = metrics.f1_score(y_test, y_pred, average='macro')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_model(self, filepath):
        """保存训练好的模型到文件"""
        if not hasattr(self, 'pipeline'):
            raise ValueError("必须先训练模型")

        dump(self.pipeline, f'{filepath}.joblib')
        print(f"模型已保存到 {filepath}.joblib")

    @staticmethod
    def load_model(filepath):
        """从文件加载模型"""
        pipeline = load(f'{filepath}.joblib')
        print(f"模型已从 {filepath}.joblib 加载")
        return pipeline


# ======================== 主程序 ========================

if __name__ == "__main__":
    # 1. 加载数据（替换为实际路径）
    loader = MNISTLoader(data_dir=r'E:\code\Pythoncode\Pycharmproject\machinelearning\data\mnist_data')
    X_train, y_train, X_test, y_test = loader.load_dataset()

    # 2. 检查数据形状
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")

    # 3. 选择优化模块
    trainer = SVMTrainer(
        use_hog=False,  # 使用HOG特征
        use_stacking=True  # 使用模型集成
    )

    # 4. 训练与评估
    print("\n开始训练...")
    trainer.train(X_train, y_train)

    print("\n评估测试集...")
    metrics_result = trainer.evaluate(X_test, y_test, show_plot=False)  # 设置为False不显示图像

    # 打印最终指标
    print("\n最终指标:")
    for name, value in metrics_result.items():
        print(f"{name}: {value:.4f}")

    # 5. 保存模型
    model_save_path = r'E:\code\Pythoncode\Pycharmproject\machinelearning\model\mnist_svm_model_test'
    trainer.save_model(model_save_path)

    # 6. 加载模型并复用
    print("\n测试模型加载与复用...")
    loaded_pipeline = SVMTrainer.load_model(model_save_path)

    # 用加载的模型做预测 - 直接输入原始图像
    sample_idx = 0  # 测试第一个样本
    sample_image = X_test[sample_idx].reshape(1, 28, 28)  # 确保正确的输入形状

    # 直接使用pipeline进行预测
    pred_label = loaded_pipeline.predict(sample_image)[0]
    true_label = y_test[sample_idx]
    print(f"样本预测: {pred_label}, 真实标签: {true_label}")

    # 可视化测试样本
    plt.imshow(X_test[sample_idx], cmap='gray')
    plt.title(f"Pred: {pred_label}, True: {true_label}")
    plt.show()