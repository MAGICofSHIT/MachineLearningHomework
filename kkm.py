# -*-coding:GBK -*-
# ������Ҫʹ�õ���ͺ���
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # ͼƬ����������ʾ
plt.rcParams['axes.unicode_minus'] = False

# �������ݼ�
load_data = load_iris()
X = load_data.data
y = load_data.target
# �鿴���ݼ���С
print('X.shape:', X.shape)
print('Y.shape:', y.shape)
print('feature_names:', load_data.feature_names)
print('target_names:', load_data.target_names)
# �鿴ǰ4������
print(X[:4, :])

# ģ��ѵ��

# �ָ����ݼ�
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20, shuffle=True)
# ����ָ���ѵ���������Լ��Ĵ�С
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)
# ����k����ģ��
model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)
# ѵ��ģ��
model.fit(X_train, y_train)
# ����ģ����ѵ���������Լ��ϵ�Ԥ��׼ȷ��
knn_train_score = model.score(X_train, y_train)
knn_test_score = model.score(X_test, y_test)
print('knn_train_score:', knn_train_score)
print('knn_test_score:', knn_test_score)

# �Ա�Ԥ��ֵ����ʵֵ
print(model.predict(X_test))
print(y_test)

train = [0 for x in range(10)]
test = [0 for x in range(10)]

for i in range(9):
    # ���ĳ�����������ѵ��ģ�ͣ�����Ч�����
    model2 = KNeighborsClassifier(n_neighbors=i + 1, weights='uniform', algorithm='auto', leaf_size=30, p=2)
    model2.fit(X_train, y_train)
    print('knn_train_score', i + 1, model2.score(X_train, y_train))
    train[i + 1] = model2.score(X_train, y_train)
    print('knn_test_score', i + 1, model2.score(X_test, y_test))
    test[i + 1] = model2.score(X_test, y_test)

plt.xlim(1, 9)
plt.ylim(0.88, 1.02)
plt.plot(train, label='ѵ����')
plt.plot(test, label='���Լ�')
plt.xlabel('kֵ')
plt.ylabel('׼ȷ��')
plt.title('��ͬkֵ��׼ȷ�ʵ�Ӱ��')
plt.legend()
plt.show()

dis_test = [0 for x in range(10)]

for i in range(9):
    # ���ĳ�����������ѵ��ģ�ͣ�����Ч�����
    model2 = KNeighborsClassifier(n_neighbors=i + 1, weights='distance', algorithm='auto', leaf_size=30, p=2)
    model2.fit(X_train, y_train)
    print('knn_train_score', i + 1, model2.score(X_train, y_train))
    print('knn_test_score', i + 1, model2.score(X_test, y_test))
    dis_test[i + 1] = model2.score(X_test, y_test)

plt.xlim(1, 9)
plt.ylim(0.88, 1.02)
plt.plot(test, label='weight=\'uniform\'')
plt.plot(dis_test, label='weight=\'distance\'')
plt.xlabel('kֵ')
plt.ylabel('׼ȷ��')
plt.title('��ͬweight����ѡȡ��׼ȷ�ʵ�Ӱ��')
plt.legend()
plt.show()
