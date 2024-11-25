import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 定义 KNN 分类的可视化函数
def plot_knn_classification_with_accuracy(n_neighbors=3):
    """
    使用 KNN 分类鸢尾花数据集，绘制决策边界，并打印准确率。

    参数：
    - n_neighbors: KNN 的邻居数，默认是 3
    """
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data[:, 2:4]  # 选择花瓣长度和宽度两个特征
    y = iris.target

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # 预测测试集
    y_pred = knn.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN 分类器 (k={n_neighbors}) 的准确率: {accuracy:.4f}")

    # 准备绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = knn.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))  # 预测平面上的每一点
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)  # 决策边界
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.Paired)  # 数据点
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(f"KNN Classification (k={n_neighbors}), Accuracy: {accuracy:.4f}")
    plt.colorbar(scatter)
    plt.show()


# 测试不同的 K 值并打印准确率
for k in range(1,11):  # 测试 k=1, 3, 5
    plot_knn_classification_with_accuracy(n_neighbors=k)
