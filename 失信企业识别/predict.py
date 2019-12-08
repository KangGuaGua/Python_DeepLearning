import numpy as np
import pandas as pd
from keras import models
from keras import layers
import matplotlib.pyplot as plt

data_train_x = pd.read_csv('train.csv')  # 读取文件.14366*178
data_label_y = pd.read_csv('train_label.csv')  # 14366*2
Pre = pd.read_csv('test.csv')   # 要预测的数据

# 用0填充nan值
Pre.fillna(0,inplace=True)
data_train_x.fillna(0,inplace=True)
print(data_train_x.head(5))
# 预处理

# 方法一，删除部分列（删除和选取列这两种方法选一种使用）
n = [2, 5, 8, 10, 11, 12, 16]  # 要删除的无关属性列
data_train_x.drop(data_train_x.columns[n], axis=1, inplace=True)  # 删除无关属性
Pre.drop(Pre.columns[n], axis=1, inplace=True)
'''
# 方法二，选取部分列
#m = [17, 18, 19, 20, 21, 51, 65, 76, 77, 87, 106, 108, 112, 117, 126, 134, 136, 148, 150]
m = [6, 18, 24, 26, 29, 38, 49, 60, 62, 65, 74, 85, 106, 108, 117, 126, 134, 148, 150, 159, 168, 176]
data_train_x = data_train_x.iloc[:,m]

'''
# 划分出训练集
x_train = data_train_x.iloc[:9000, :].values  # 直接读取的数据为DataFrame类型，转换为ndarray类型处理
y_train = data_label_y.iloc[:9000, 1].values  # 9000*1为训练集

# 划分出验证集和测试集
x_val = data_train_x.iloc[9000:12000, :].values  # 3000*178表示验证集
y_val = data_label_y.iloc[9000:12000, 1].values
x_test = data_train_x.iloc[12000:, :].values   # 2366个用来测试
y_test = data_label_y.iloc[12000:, 1].values

# 预测
Pred = Pre.iloc[:].values
#Pred = Pre.iloc[:,m].values
x_pre = Pred    # 要预测的企业的信息
y_pre = np.zeros((x_pre.shape[0], 1))  # 需要预测的值

sum = Pred.shape[0]   # sum表示要预测的个数
x_ID = Pre.iloc[:,0].values.reshape((sum,1)) #将预测的企业ID取出

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape,x_test.shape, x_pre.shape, data_train_x.shape,)
print(Pred.shape)


# 建立模型
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(8, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(x_train,y_train,epochs=20,batch_size=128,validation_data=(x_val,y_val))

# 结果输出
results = model.evaluate(x_test, y_test)
y_pre = model.predict(x_pre)
pred = np.hstack((x_ID, y_pre))    # 将x，y合成提交文件的格式
print(results)
print(pred)
np.savetxt('prediction.csv', pred, fmt='%2f', delimiter=',', header='ID,Label')  # 将结果保存到文件

# 绘制训练损失和验证损失
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs=range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制训练精度和验证精度
plt.clf()
acc=history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()