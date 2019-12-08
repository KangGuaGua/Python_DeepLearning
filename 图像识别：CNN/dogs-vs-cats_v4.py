# 使用数据增强的特征提取，扩展conv_base模型，然后在输入数据上端到端地运行模型
# 本方法计算代价高，只在有GPU的情况下才能尝试运行
# 模型的行为和层类似，可以向Sequential模型中添加一个模型(比如conv_base)，就像添加一个层一样

from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))  # 输入图像的形状

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
base_dir = 'D:/PyCharm/Project/DeepLearning/图像识别：CNN/cats_and_dogs_small'
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

# 展开为（samples，8192）,自定义密集连接分类器
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

# 构建网络
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())  # 输出此时模型的架构

# 在编译和训练模型之前，一定要冻结卷积基，即在训练过程中使一个或多个层保持权重不变。
# 若不冻结，卷积基之前学到的表示将会在训练过程中被修改。因为其上添加的Dense层是随机初始化的，
# 所以非常大的权重更新将会在网络中传播，对之前学到的表示造成很大破坏。
print('冻结conv_base之前的要训练权重数量：',len(model.trainable_weights))
conv_base.trainable = False  # 冻结卷积基
print('冻结之后的conv_base要训练的权重数量:',len(model.trainable_weights))
# 冻结以后，只有添加的两个Dense层的权重才会被训练。一共为4个，每层两个(主权重矩阵和偏置向量)
# 为了让修改生效，必须先编译模型。如果在编译之后修改了权重的trainable属性，应该重新编译模型，否则这些修改将被忽略

# 利用冻结的卷积基端到端地训练模型
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)  # 不能增强验证数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data = validation_generator,
    validation_steps=50)

# 此次的结果有96%的精度，和之前相比有很大的提高