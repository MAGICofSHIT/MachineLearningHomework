# -*-coding:GBK -*-
# 导入需要使用的类和函数
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片标题中文显示
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
load_data = load_iris()
X = load_data.data
y = load_data.target
# 查看数据集大小
print('X.shape:', X.shape)
print('Y.shape:', y.shape)
print('feature_names:', load_data.feature_names)
print('target_names:', load_data.target_names)
# 查看前4笔数据
print(X[:4, :])

# 模型训练

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle=True)
# 输出分割后的训练集、测试集的大小
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)
# 生成k近邻模型
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)
# 训练模型
model.fit(X_train, y_train)
# 看看模型在训练集、测试集上的预测准确率
knn_train_score = model.score(X_train, y_train)
knn_test_score = model.score(X_test, y_test)
print('knn_train_score:', knn_train_score)
print('knn_test_score:', knn_test_score)

# 对比预测值与真实值
print(model.predict(X_test))
print(y_test)

train = [0 for x in range(10)]
test = [0 for x in range(10)]

for i in range(9):
    # 更改超参数，重新训练模型，看看效果如何
    model2 = KNeighborsClassifier(n_neighbors=i + 1, weights='uniform', algorithm='auto', leaf_size=30, p=2)
    model2.fit(X_train, y_train)
    print('knn_train_score', i + 1, model2.score(X_train, y_train))
    train[i + 1] = model2.score(X_train, y_train)
    print('knn_test_score', i + 1, model2.score(X_test, y_test))
    test[i + 1] = model2.score(X_test, y_test)

plt.xlim(1, 9)
plt.ylim(0.88, 1.02)
plt.plot(train, label='训练集')
plt.plot(test, label='测试集')
plt.xlabel('k值')
plt.ylabel('准确率')
plt.title('不同k值对准确率的影响')
plt.legend()
plt.show()

dis_test = [0 for x in range(10)]

for i in range(9):
    # 更改超参数，重新训练模型，看看效果如何
    model2 = KNeighborsClassifier(n_neighbors=i + 1, weights='distance', algorithm='auto', leaf_size=30, p=2)
    model2.fit(X_train, y_train)
    print('knn_train_score', i + 1, model2.score(X_train, y_train))
    print('knn_test_score', i + 1, model2.score(X_test, y_test))
    dis_test[i + 1] = model2.score(X_test, y_test)

plt.xlim(1, 9)
plt.ylim(0.88, 1.02)
plt.plot(test, label='weight=\'uniform\'')
plt.plot(dis_test, label='weight=\'distance\'')
plt.xlabel('k值')
plt.ylabel('准确率')
plt.title('不同weight参数选取对准确率的影响')
plt.legend()
plt.show()
