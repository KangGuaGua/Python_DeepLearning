# *************************************************************************
# 在上一个基础上修改轮次为500次，并且画出每个轮次中MAE的平均值图像，其余部分与之前代码相同
# *************************************************************************

from keras.datasets import boston_housing  # 加载波士顿房价数据，单位为千美元
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 构建网络。
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# K折验证
import numpy as np
k = 4
num_val_samples = len(train_data)//k
num_epochs = 500
all_mae_histoties = []

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
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=2)
    # verbose=0表示在控制台没有任何输出，=1表示显示进度条，=2表示为每个epoch输出一行记录
    mae_history = history.history['val_mae']
    all_mae_histoties.append(mae_history)

# 计算每个轮次中所有折MAE的平均值
average_mae_history = [
    np.mean([x[i] for x in all_mae_histoties]) for i in range(num_epochs)]

# 绘制验证分数图像
import  matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 由上图难看出规律。所以重新画图：a.删除前10个数据点。
# b.将每个数据点替换为前面数据点的指数移动平均值，以得到光滑的曲线。
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()