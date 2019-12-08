# **************************************************
# 仅使用数据增强，只靠从头开始训练自己的卷积神经网络再想提高精度十分困难，因为可用的数据太少。
# 想进一步提高精度，需要使用预训练的ＣＮＮ
# 这次仅使用特征提取，未使用数据增强。
# 在数据集上运行卷积基，将输出保存成硬盘的Numpy数组，然后用这个数据作为输入，输入到独立的
# 密集连接分类器中。这种方法速度快，计算代价低。每个输入图像只需运行一次卷积基。
# **************************************************
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))  # 输入图像的形状
# weights 指定模型初始化的权重检查点
# include_top指定模型最后是否包含密集连接分类器。默认情况下ImageNet输出1000个类别
print(conv_base.summary())
# 可以看出最后的特征图形状为（4， 4， 512）

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
base_dir = 'D:/PyCharm/Project/Python_DeepLearning/图像识别：CNN/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i +=1
        if i*batch_size>=sample_count:
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 目前提取的特征形状为(samples, 4, 4, 512).需要将其输入到密集连接分类器中
# 首先将其展开为（samples，8192）,自定义密集连接分类器
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

# 构建网络
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

# 绘制训练期间的损失曲线和精度曲线
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 从结果可以看出，虽然使用了dropout，但模型还是一开始就过拟合，这是因为数据集较小，没有使用数据增强