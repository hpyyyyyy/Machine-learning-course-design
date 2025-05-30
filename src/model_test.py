# model_diagnosis.py

import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from joblib import load
from sklearn import metrics
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn import svm
import struct
import matplotlib as mpl
from sklearn.ensemble import StackingClassifier

# 设置全局字体为支持英文的DejaVu Sans
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# ======================== 配置参数 ========================

# 模型路径
MODEL_PATH = r'E:\code\Pythoncode\Pycharmproject\machinelearning\model\mnist_svm_model.joblib'

# 数据集路径
DATA_DIR = r'E:\code\Pythoncode\Pycharmproject\machinelearning\data\mnist_data'

# 输出目录
OUTPUT_DIR = r'E:\code\Pythoncode\Pycharmproject\machinelearning\data\diagnosis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== MNIST数据加载模块 ========================

class MNISTLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_images(self, filename):
        """Load MNIST image binary file"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_labels(self, filename):
        """Load MNIST label binary file"""
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    def load_dataset(self):
        """Load full training and test datasets"""
        X_train = self.load_images('train-images-idx3-ubyte')
        y_train = self.load_labels('train-labels-idx1-ubyte')
        X_test = self.load_images('t10k-images-idx3-ubyte')
        y_test = self.load_labels('t10k-labels-idx1-ubyte')
        return X_train, y_train, X_test, y_test


# ======================== HOG特征提取器 ========================

class HogTransformer:
    """HOG Feature Extractor (same as in training)"""

    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def get_params(self, deep=True):
        """Get parameters for this estimator (required for sklearn compatibility)"""
        return {
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block
        }

    def transform(self, X):
        from skimage.feature import hog
        hog_features = []
        for img in X:
            if len(img.shape) == 3:
                img = img.squeeze()
            feat = hog(img,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       visualize=False)
            hog_features.append(feat)
        return np.array(hog_features)


# ======================== 模型诊断分析模块 ========================

class ModelDiagnoser:
    def __init__(self, pipeline, X_train, y_train, X_test, y_test):
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.diagnosis_dir = OUTPUT_DIR
        os.makedirs(self.diagnosis_dir, exist_ok=True)

        # Check model type
        self.use_hog = any("hog" in step[0] for step in pipeline.steps)
        self.use_stacking = 'model' in pipeline.named_steps and isinstance(
            pipeline.named_steps['model'], StackingClassifier)

    def learning_curve_analysis(self):
        """Learning curve analysis - no retraining needed"""
        print("\nStarting learning curve analysis...")

        # Create training subsets of different sizes
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs = [int(len(self.X_train) * size) for size in train_sizes]

        train_scores = []
        test_scores = []

        # Evaluate on different training set sizes
        for size in train_sizes_abs:
            X_subset = self.X_train[:size]
            y_subset = self.y_train[:size]

            # Training set evaluation
            train_pred = self.pipeline.predict(X_subset)
            train_acc = metrics.accuracy_score(y_subset, train_pred)
            train_scores.append(train_acc)

            # Test set evaluation
            test_pred = self.pipeline.predict(self.X_test)
            test_acc = metrics.accuracy_score(self.y_test, test_pred)
            test_scores.append(test_acc)

            print(f"Training samples: {size}, Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

        # Visualize learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_scores, 'o-', color="r", label="Training accuracy")
        plt.plot(train_sizes_abs, test_scores, 'o-', color="g", label="Test accuracy")

        # Add reference lines
        final_train_acc = train_scores[-1]
        final_test_acc = test_scores[-1]
        plt.axhline(y=final_train_acc, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=final_test_acc, color='g', linestyle='--', alpha=0.3)

        # Annotate accuracy gap
        gap = final_train_acc - final_test_acc
        gap_text = f"Accuracy gap: {gap:.4f}"
        plt.annotate(gap_text, xy=(train_sizes_abs[-1], (final_train_acc + final_test_acc) / 2),
                     xytext=(-100, 30), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        plt.title("Learning Curve Analysis")
        plt.xlabel("Number of Training Samples")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.grid(True)

        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        curve_path = os.path.join(self.diagnosis_dir, f'learning_curve_{timestamp}.png')
        plt.savefig(curve_path)
        plt.close()
        print(f"Learning curve saved to: {curve_path}")

        # Analyze results
        if gap > 0.1:
            diagnosis = "Severe overfitting: Large gap between train and test accuracy"
        elif gap > 0.05:
            diagnosis = "Moderate overfitting: Train accuracy significantly higher than test"
        elif gap < 0.01:
            diagnosis = "Good fit: Train and test performance are close"
        elif final_test_acc < 0.85:
            diagnosis = "Underfitting: Poor performance on both train and test sets"
        else:
            diagnosis = "Slight overfitting: Consider adjusting regularization"

        return {
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "accuracy_gap": gap,
            "diagnosis": diagnosis,
            "curve_path": curve_path
        }

    def feature_importance_analysis(self):
        """Feature importance analysis (for ensemble models)"""
        if not self.use_stacking:
            print("Warning: Feature importance analysis only available for ensemble models")
            return None

        # Get final model (logistic regression)
        final_model = self.pipeline.named_steps['model'].final_estimator_

        # Get feature importances (logistic regression coefficients)
        if hasattr(final_model, 'coef_'):
            feature_importances = np.mean(np.abs(final_model.coef_), axis=0)
        else:
            print("Final model doesn't have feature importances")
            return None

        # Visualize feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.title("Feature Importance Analysis")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")

        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        importance_path = os.path.join(self.diagnosis_dir, f'feature_importance_{timestamp}.png')
        plt.savefig(importance_path)
        plt.close()
        print(f"Feature importance plot saved to: {importance_path}")

        return {
            "importances": feature_importances.tolist(),
            "importance_path": importance_path
        }

    def decision_boundary_analysis(self, num_samples=1000):
        """Decision boundary visualization (using PCA)"""
        print("\nPerforming decision boundary analysis...")

        # Random sampling to reduce computation
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        X_sample = self.X_train[indices]
        y_sample = self.y_train[indices]

        # Extract features before prediction
        if self.use_hog:
            # Use HOG features
            hog_params = self.pipeline.named_steps['hog'].get_params()
            hog_transformer = HogTransformer(**hog_params)
            X_features = hog_transformer.transform(X_sample)
        else:
            # Use raw features
            X_features = X_sample.reshape(X_sample.shape[0], -1)

        # Standardize features
        scaler = self.pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X_features)

        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Train a small SVM for visualization
        svm_model = svm.SVC(kernel='rbf', C=1, gamma='scale')
        svm_model.fit(X_pca, y_sample)

        # Create mesh grid for decision boundary
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Predict entire grid
        Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(12, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.tab10)

        # Plot sample points (10 per class)
        for i in range(10):
            idx = np.where(y_sample == i)[0]
            plt.scatter(X_pca[idx[:10], 0], X_pca[idx[:10], 1],
                        label=str(i), s=30, alpha=0.8)

        plt.title("Decision Boundary Visualization (PCA Reduced)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Digit Class", loc='best')

        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        boundary_path = os.path.join(self.diagnosis_dir, f'decision_boundary_{timestamp}.png')
        plt.savefig(boundary_path)
        plt.close()
        print(f"Decision boundary plot saved to: {boundary_path}")

        return boundary_path

    def prediction_confidence_analysis(self):
        """Prediction confidence analysis"""
        # Get test set prediction probabilities
        if hasattr(self.pipeline, 'predict_proba'):
            proba = self.pipeline.predict_proba(self.X_test)
            confidences = np.max(proba, axis=1)
        else:
            # For models without probability, use decision function
            decision = self.pipeline.decision_function(self.X_test)
            confidences = np.max(decision, axis=1)
            # Normalize to [0,1] range
            confidences = (confidences - np.min(confidences)) / (np.max(confidences) - np.min(confidences))

        # Calculate confidence for correct and incorrect predictions
        preds = self.pipeline.predict(self.X_test)
        correct = preds == self.y_test
        correct_conf = confidences[correct]
        error_conf = confidences[~correct]

        # Visualize confidence distribution
        plt.figure(figsize=(10, 6))
        plt.hist(correct_conf, bins=30, alpha=0.5, label='Correct Predictions', color='green')
        plt.hist(error_conf, bins=30, alpha=0.5, label='Incorrect Predictions', color='red')
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Number of Samples")
        plt.legend()

        # Save image
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        conf_path = os.path.join(self.diagnosis_dir, f'confidence_dist_{timestamp}.png')
        plt.savefig(conf_path)
        plt.close()
        print(f"Confidence distribution plot saved to: {conf_path}")

        # Calculate metrics
        mean_correct_conf = np.mean(correct_conf) if len(correct_conf) > 0 else 0
        mean_error_conf = np.mean(error_conf) if len(error_conf) > 0 else 0

        return {
            "mean_correct_confidence": float(mean_correct_conf),
            "mean_error_confidence": float(mean_error_conf),
            "confidence_path": conf_path
        }

    def full_diagnosis(self, include_learning_curve=True):
        """Complete diagnosis workflow"""
        print("\n" + "=" * 50)
        print("Starting Model Overfitting/Underfitting Diagnosis")
        print("=" * 50)

        report = {}

        # 1. Learning curve analysis
        if include_learning_curve:
            curve_result = self.learning_curve_analysis()
            report["learning_curve"] = curve_result
        else:
            print("Skipping learning curve analysis...")

        # 2. Feature importance analysis
        feature_result = self.feature_importance_analysis()
        if feature_result:
            report["feature_importance"] = feature_result

        # 3. Decision boundary analysis
        boundary_path = self.decision_boundary_analysis()
        report["decision_boundary"] = boundary_path

        # 4. Prediction confidence analysis
        conf_result = self.prediction_confidence_analysis()
        report["confidence_analysis"] = conf_result

        # 5. Comprehensive diagnosis report
        if include_learning_curve:
            report["diagnosis"] = curve_result["diagnosis"]

            # Print diagnosis results
            print("\nDiagnosis Results:")
            print(f"- Training accuracy: {curve_result['final_train_acc']:.4f}")
            print(f"- Test accuracy: {curve_result['final_test_acc']:.4f}")
            print(f"- Accuracy gap: {curve_result['accuracy_gap']:.4f}")
            print(f"- Diagnosis: {curve_result['diagnosis']}")

            # Recommendations
            print("\nRecommendations:")
            if "overfitting" in curve_result["diagnosis"].lower():
                print("1. Increase regularization strength (reduce C value)")
                print("2. Add data diversity (data augmentation)")
                print("3. Reduce model complexity (e.g., use single model)")
                print("4. Add dropout or feature selection")
            elif "underfitting" in curve_result["diagnosis"].lower():
                print("1. Increase model complexity (e.g., deeper ensemble)")
                print("2. Improve feature engineering (e.g., add HOG features)")
                print("3. Reduce regularization strength (increase C value)")
                print("4. Increase training iterations or data size")
            else:
                print("Model has good fit, ready for deployment or further optimization")

        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.diagnosis_dir, f'diagnosis_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"\nDiagnosis report saved to: {report_path}")

        return report


# ======================== 主程序 ========================

if __name__ == "__main__":
    # 1. Load dataset
    print("Loading MNIST dataset...")
    loader = MNISTLoader(DATA_DIR)
    X_train, y_train, X_test, y_test = loader.load_dataset()
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 2. Load model
    print(f"\nLoading model: {MODEL_PATH}")
    try:
        pipeline = load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit(1)

    # 3. Create diagnoser
    diagnoser = ModelDiagnoser(pipeline, X_train, y_train, X_test, y_test)

    # 4. Perform full diagnosis
    # Parameter controls whether to include learning curve analysis
    diagnoser.full_diagnosis(include_learning_curve=True)