import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
# 从源文件加载数据，并输出查看数据的各项特征
path = r'D:\edge\ML_DataSet\USA_Housing.csv'
lines = np.loadtxt(path, delimiter=',', dtype='str')
header = lines[0]
lines = lines[1:].astype(float)
print('数据特征：', ', '.join(header[:-1]))
print('数据标签：', header[-1])
print('数据总条数：', len(lines))

# 划分训练集与测试集
ratio = 0.8
split = int(len(lines) * ratio)
np.random.seed(0)
lines = np.random.permutation(lines)
train, test = lines[:split], lines[split:]

# 数据归一化
scaler = StandardScaler()
scaler.fit(train)  # 只使用训练集的数据计算均值和方差
train = scaler.transform(train)
test = scaler.transform(test)

# 划分输入和标签
x_train, y_train = train[:, :-1], train[:, -1].flatten()
x_test, y_test = test[:, :-1], test[:, -1].flatten()

# sklearn内置模型
# lin = LinearRegression()
# lin.fit(x_train, y_train)
# print("回归系数：", lin.coef_, lin.intercept_)
# y_pred = lin.predict(x_test)
# loss = np.sqrt(np.square(y_test - y_pred).mean())
# print(loss)

# batch gradient descent
def batch_generator(x, y, batch_size, shuffle=True ):
    batch_count = 0
    if shuffle:
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(x))
        if start_index >= end_index:
            break
        batch_count += 1
        yield x[start_index:end_index], y[start_index:end_index]

def SGD(num_epoch, learning_rate, batch_size):
    X = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)

    theta = np.random.normal(size=X.shape[1])

    train_losses= []
    test_losses = []
    for i in range(num_epoch):
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0
        for x_batch, y_batch in batch_g:
            # compute the gradeint
            grad = x_batch.T @ (x_batch @ theta - y_batch)
            # renwe theta
            theta = theta - learning_rate * grad / len(x_batch)
            # compute loss
            train_loss += np.square(x_batch @ theta - y_batch).sum()
        train_loss = np.sqrt(train_loss / len(X))
        train_losses.append(train_loss)
        # 训练完一个epoch后，计算测试集误差
        test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
        test_losses.append(test_loss)
    return theta, train_losses, test_losses

# 设置迭代次数，学习率与批量大小
num_epoch = 20
learning_rate = 0.01
batch_size = 32
# 设置随机种子
np.random.seed(0)

_, train_losses, test_losses = SGD(num_epoch, learning_rate, batch_size)

# 将损失函数关于运行次数的关系制图，可以看到损失函数先一直保持下降，之后趋于平稳
plt.plot(np.arange(num_epoch), train_losses, color='blue',
    label='train loss')
plt.plot(np.arange(num_epoch), test_losses, color='red',
    ls='--', label='test loss')
# 由于epoch是整数，这里把图中的横坐标也设置为整数
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()
