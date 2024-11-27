import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标题
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, [2, 3]]  # 选择花瓣长度和宽度两个特征
y = iris.target

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 初始化 KNN 模型
knn_model = KNeighborsClassifier(n_neighbors=6, p=2)  # 设置近邻数为 6

# 训练模型
knn_model.fit(X_train, y_train)

# 测试模型并计算准确率
test_accuracy_knn = knn_model.score(X_test, y_test) * 100
train_accuracy_knn = knn_model.score(X_train, y_train) * 100
print(f"KNN 测试集准确率: {test_accuracy_knn:.6f}%")
print(f"KNN 训练集准确率: {train_accuracy_knn:.6f}%")

# 准备绘制决策边界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格上的每一点
Z_knn = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_knn = Z_knn.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(12, 7))
plt.contourf(xx, yy, Z_knn, alpha=0.8, cmap=plt.cm.Paired)

# 绘制训练集数据点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k", s=80,
            marker='o', cmap=plt.cm.Paired, label="训练集数据")

# 绘制测试集数据点
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="k", s=120,
            marker='^', cmap=plt.cm.Paired, label="测试集数据")

# 图例和标签
plt.xlabel("花瓣长度(cm)")
plt.ylabel("花瓣宽度(cm)")
plt.title(f"KNN 分类, 训练集准确率: {train_accuracy_knn:.6f}%，测试集准确率: {test_accuracy_knn:.6f}%")
plt.legend(loc="upper left", fontsize=12)
plt.colorbar()  # 添加颜色条
plt.savefig('./Pictures/KNN_Classification.png')

# 存储不同 k 值下的准确率
k_values = range(1, 16)  # K 值从 1 到 15
train_accuracies = []
test_accuracies = []

# 遍历不同的 k 值
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k, p=2)
    knn_model.fit(X_train, y_train)

    # 计算训练集和测试集的准确率
    train_accuracy = knn_model.score(X_train, y_train) * 100
    test_accuracy = knn_model.score(X_test, y_test) * 100

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(f"k={k}的KNN模型: 训练集准确率={train_accuracy:.6f}%，测试集准确率={test_accuracy:.6f}%")

# 绘制 K 值与准确率的折线图
plt.figure(figsize=(12, 7))
plt.plot(k_values, train_accuracies, label="训练集准确率", marker='o', linestyle='-')
plt.plot(k_values, test_accuracies, label="测试集准确率", marker='^', linestyle='--')
plt.xlabel("k 值")
plt.ylabel("准确率 (%)")
plt.title("不同 k 值下 KNN 模型的分类准确率")
plt.legend(loc="best")
plt.grid(True)
plt.savefig('./Pictures/KNN_Accuracy_vs_K.png')
