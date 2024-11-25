# 导入需要的函数和类
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图片标题中文显示
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
load_data = fetch_california_housing()
X = load_data.data
y = load_data.target

# 输出数据集的维度
print(X.shape)
print(y.shape)
# 输出数据集包含的特征
print(load_data.feature_names)
# 输出第一笔数据
print(X[0, :])

# 模型训练
# 分割数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y
                                                                    , test_size=0.2, random_state=20, shuffle=True)
# 输出分割后训练集、测试集的维度
# print('X_train.shape:', X_train.shape)
# print('X_test.shape:', X_test.shape)
# print('y_train.shape:', y_train.shape)
# print('y_test.shape:', y_test.shape)

# 生成线性回归模型
model = LinearRegression()
# 用训练集的数据训练模型
model.fit(X_train, y_train)
# 用训练好的模型预测测试集中前两笔数据的房价，与它们的真实值做对比输出
y_predict = model.predict(X_test[:2, ])
print('y_realValue:', y_predict)
print('y_realValue:', y_test[:2, ])
# 测试模型在训练集上的得分
trainData_score = model.score(X_train, y_train)
# 测试模型在测试集上的得分
testData_score = model.score(X_test, y_test)
# 将得分输出
print('trainData_score:', trainData_score)
print('testData_score:', testData_score)

# 输出训练后的多元线性回归模型的参数的值
print(model.coef_)
print(model.intercept_)

# 模型优化
# 导入相关类和函数
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# 编写函数，将线性回归模型与多项式结合
def polynomial_LinearRegression_model(degree=1):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    model = make_pipeline(StandardScaler(), LinearRegression())
    # model = LinearRegression(normalize=True)
    # 用一个管道将线性回归模型和多项式串起来
    pipeline_model = Pipeline([("polynomial_features", poly), ("linear_regression", model)])
    return pipeline_model


# 生成包含二阶多项式的线性回归模型
model = polynomial_LinearRegression_model(degree=2)

# 训练模型
model.fit(X_train, y_train)
# 计算模型在训练集、测试集上的得分
trainData_score = model.score(X_train, y_train)
testData_score = model.score(X_test, y_test)
# 输出得分
print('trainData_score:', trainData_score)
print('testData_score:', testData_score)

# 生成包含三阶多项式的线性回归模型
model = polynomial_LinearRegression_model(degree=3)
model.fit(X_train, y_train)
trainData_score = model.score(X_train, y_train)
testData_score = model.score(X_test, y_test)
print('trainData_score:', trainData_score)
print('testData_score:', testData_score)

# 学习曲线
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np


# 编写函数，画出学习曲线图
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1
                        , train_sizes=np.linspace(0.1, 1.0, 5)):
    # 图像标题
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    # x轴、y轴标题
    plt.xlabel("训练样例数")
    plt.ylabel("评估得分")
    # 获取训练集大小，训练得分集合，测试得分集合
    train_sizes, train_scores, test_scores = learning_curve(estimator
                                                            , X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # 计算均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 背景设置为网格线
    plt.grid()
    # 画出模型得分的均值
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r'
             , label='训练集')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g'
             , label='交叉验证集')
    # 显示图例
    plt.legend(loc='best')
    return plt


# 分别画出阶数为1、2、3时模型的学习曲线
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
# 交叉验证类进行十次迭代，测试集占0.3，其余的都是训练集
titles = '学习曲线 (阶数={0})'
# 多项式的阶数
degrees = [1, 2, 3]
# 设置画布大小，dpi是每英寸的像素点数
plt.figure(figsize=(18, 4), dpi=200)
# 循环三次
for i in range(len(degrees)):
    # 下属三张画布，对应编号为i+1
    # plt.subplot(1, 3, i + 1)
    # 开始绘制曲线
    plot_learning_curve(polynomial_LinearRegression_model(degrees[i])
                        , titles.format(degrees[i]), X, y, ylim=(0.01, 1.01), cv=cv)
    # 显示
    plt.show()
