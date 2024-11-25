# 导入必要的库
from sklearn.datasets import fetch_california_housing  # 加载加州房价数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split  # 数据集拆分工具
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import r2_score  # R^2 评估指标
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from california import load_data


# 定义函数，将多项式特征和线性回归连接成管道
def polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    """
    使用多项式回归预测并评估模型性能。

    参数：
    - X_train, X_test: 特征的训练集和测试集
    - y_train, y_test: 目标变量的训练集和测试集
    - degree: 多项式的阶数，默认是2

    返回：
    - r2: 模型在测试集上的 R^2 分数
    - y_pred: 测试集的预测值
    """
    # 创建管道
    pipeline = Pipeline([
        ("scaler", StandardScaler()),  # 标准化处理
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),  # 多项式特征
        ("linear_regression", LinearRegression())  # 线性回归模型
    ])

    # 在训练集上训练模型
    pipeline.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = pipeline.predict(X_test)

    # 计算 R^2 分数
    r2 = r2_score(y_test, y_pred)

    return r2, y_pred


# 加载加州房价数据集
# data = fetch_california_housing(as_frame=True)
data = load_iris()
df = data.frame  # 获取数据集的 DataFrame 格式

# 特征和目标变量
X = df.drop(columns=["MedHouseVal"])  # 特征：去掉目标列
y = df["MedHouseVal"]  # 目标变量：房屋中位价

# 将数据集拆分为训练集和测试集（8:2比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=420)

# 初始化线性回归模型
model = LinearRegression()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 使用 R^2 评估模型性能
r2 = r2_score(y_test, y_pred)

# 打印 R^2 结果
print(f"模型的 R^2 分数为: {r2:.4f}")

# 调用函数，测试不同的多项式阶数
for degree in range(1, 4):  # 尝试 1 到 3 阶
    r2, _ = polynomial_regression(X_train, X_test, y_train, y_test, degree=degree)
    print(f"多项式回归模型 (阶数={degree}) 的 R^2 分数为: {r2:.4f}")

# 查看数据集的列名和前几行数据
# print(df.columns)  # 显示所有的列名
# print(data.feature_names)  # 查看特征名称
# print(data.target_names)   # 查看目标名称
# print(df.head())    # 显示前几行数据
