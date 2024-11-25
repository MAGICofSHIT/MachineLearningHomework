import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 定义多项式逻辑回归的可视化函数
def plot_polynomial_classification_with_accuracy(degree=2):
    """
    使用多项式逻辑回归分类鸢尾花数据集，绘制决策边界，并打印准确率。

    参数：
    - degree: 多项式的阶数，默认是2
    """
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data[:, 2:4]  # 选择花瓣长度和宽度两个特征
    y = iris.target

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

    # 创建管道：标准化 -> 多项式特征 -> 逻辑回归
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # 标准化
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),  # 多项式特征
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))  # 逻辑回归
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 测试模型并计算准确率
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"多项式逻辑回归 (degree={degree}) 的准确率: {accuracy:.4f}")

    # 准备绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])  # 预测平面上的每一点
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # 决策边界
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)  # 数据点
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(f"Polynomial Logistic Regression (degree={degree}), Accuracy: {accuracy:.4f}")
    plt.colorbar(scatter)
    plt.show()


# 测试不同阶数的多项式并打印准确率
for degree in range(1, 4):  # 测试 1 到 3 阶
    plot_polynomial_classification_with_accuracy(degree=degree)
