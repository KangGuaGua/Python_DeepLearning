from keras.datasets import boston_housing  # 加载波士顿房价数据，单位为千美元
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print('The shape of Train sets and Test sets', train_data.shape, test_data.shape)
print('train_targets', train_targets)

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean  # 标准化测试集也要使用来自训练集上的均值和标准差。不能使用在测试集上计算得到的结果。
test_data /= std

# 构建网络。因为需要将同一个模型多次实例化，所以用一个函数来构建模型
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # 最后一层只有一个单元，没有激活，是一个线性层。这是标量回归(标量回归是预测单一连续值的回归)的典型设置。
    # 添加激活函数将会限制输出范围。而纯线性的可以预测任意范围内的值。
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # 损失函数用的mse(均方误差，mean squared error)，预测值与目标值之差的平方。这是回归常用的损失函数
    # 指标用的是mae(平均绝对误差，mean absolute error)，预测值与目标值之差的绝对值。
    return model

# *************************************************************************************
# 使用K折验证来验证网络。因为训练集样本太少，传统划分方法的验证分数会有很大波动。验证集的划分方式会造成
# 验证分数有很大的方差。所以采用K折交叉验证。
# K折验证：将数据划分为K个分区，实例化K个相同的模型，将每个模型在K-1个分区上训练，并在剩下的一个分区进行评估。
# 模型的验证分数等于K个验证分数的平均值
# *************************************************************************************
# K折验证
import numpy as np
k = 4
num_val_samples = len(train_data)//k  # 划分以后每一个子集的长度。“//”表示整数除法，只返回结果的整数部分
num_epochs = 100
all_scores = []

for i in range(k):
    print('processing fold #',i)
    # 划分出验证集
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]  # 第k个分区的数据
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    # 划分出训练集
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    # verbose=0表示在控制台没有任何输出，=1表示显示进度条，=2表示为每个epoch输出一行记录
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# 输出验证分数结果以及平均得分
print(all_scores,np.mean(all_scores))
# 得到平均mae为2.3左右，说明结果偏差为2-3千美元，相对总价10000-50000美元来说比较大。所以可以增加epoch大小。
# 改进：增加轮次，并且画出MAE平均值变化图。见Price_of_house_v2.py