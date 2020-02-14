import cv2
import os
from keras.models import Sequential  # keras里面用于创建深度学习模型的包
from keras.layers import Dense, Dropout, Activation, Flatten  # 创建输出层、正则化函数、激活函数、平坦层
from keras.layers import Conv2D, MaxPooling2D  # 卷积层、最大池化层
from sklearn.preprocessing import OneHotEncoder  # 引入独热编码魔块，将特征转化为独热编码模式
import numpy as np
import glob  # 负责筛选，我们用它筛选jpg结尾的文件
from skimage import io, transform  # 负责图像的读取和转换
from sklearn import svm
from sklearn import datasets
import pickle
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w = 128
h = 128
c = 3  # 指定处理之后的图像大小和原始图像储存位置，3代表彩色图片
path = 'D:/Python 文件/基于卷积神经网络实现景区精准识别实验指导V2.5/img-classified/'


def One_Hot_Label_Encoding(labels):  # 利用独热编码模块构建自己的独热编码函数
    label = labels  # 将调用函数时输入的特征放到label变量中
    enc = OneHotEncoder()  # 创建一个独热编码对象
    enc.fit(label)  # 利用创建的独热编码对象配定给定的特征
    one_hot_labels = enc.transform(label).toarray()  # 利用匹配好的模型对特征进行转换，转换后的特征就是模型可以接受的输入格式
    return one_hot_labels  # 将转换后的特征进行返回


def read_img(path):  # 遍历文件夹，拿到子文件的名称（即图像类别，也是label的标签
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []  # 定义新的图像列表和标签列表，后面通过循环一次添加内容
    labels = []
    for idx, folder in enumerate(cate):  # 从cate中拿到每张图像对应的文件夹，idx是序号，实际上用不到，但是出于格式原因
        for im in glob.glob(folder + '/*.jpg'):  # 遍历拿到每张jpg结尾的图像
            img = io.imread(im)  # 读取图像
            img = transform.resize(img, (w, h))  # 转换图像的大小
            imgs.append(img)  # 将转换好的图像加入图像列表
            label = int(folder.split('/')[-1])  # 拿到当前图像对应的文件夹，前面获取的folder包含上一级目录，在这里进行切分
            labels.append(label)  # 将得到的特征添加进标签列表
    labels = np.array(labels)  # 将标签列表转换为numpy数据格式
    newlabels = labels[:, np.newaxis]  # 将标签转换为二维的，新增一个空的维度
    one_hot_labels = One_Hot_Label_Encoding(newlabels)  # 调用前面的函数，将标签进行独热编码转换
    return np.asarray(imgs, np.float32), one_hot_labels  # 把图像列表转换为numpy数组，然后标签数组也转换为了numpy数组，再将两者返回给调用者


data, labels = read_img(path)  # 调用前面的方法，将处理好的图像数据和标签数据
print('shape of data:', data.shape)  # 查看图像数据和标签数据的形状
print('shape of label:', labels.shape)

data_normalize = data / 255.0  # 因为图像数据在0-255之前，除于255可以让数值变小，计算更方便
num_example = data_normalize.shape[0]  # 拿到数据集样本的个数
arr = np.arange(num_example)  # 产生对应样本个数的数组
np.random.shuffle(arr)  # 随机打乱上述数组的顺序
data_normalize = data_normalize[arr]  # 按照刚才打乱数组的顺序对图像数据样本进行打乱
labels = labels[arr]  # 让标签也安装相同顺序打乱，以便和图像数据对得上

# 训练集和测试集分解
ratio = 0.8  # 指定分隔比例
s = np.int(num_example * ratio)  # 保证样本数量是整数
x_train = data_normalize[:s]  # 按照刚才计算得到的数量，从前面取对应数量的图像数据
y_train = labels[:s]  # 同理，取对应标签
x_test = data_normalize[s:]  # 取剩下部分作为训练集
y_test = labels[s:]

model = Sequential()  # 创建一个深度学习模型
model.add(Conv2D(filters=32, kernel_size=(9, 9),    # 在模型中加入卷积输入层，创建32个卷积输入核，卷积核大小为9*9
                 input_shape=(w, h, 3),             # 指定模型输入数据的大小
                 activation='relu',                 # 指定激活函数
                 padding='same'))                   # 表示处理后的图像与原图像大小一致
model.add(Dropout(rate=0.25))                       # 加入正则化层，随机丢弃25%的神经元
model.add(MaxPooling2D(pool_size=(4, 4)))           # 加入池化层，将4*4的像素缩减为一个
model.add(Conv2D(filters=64, kernel_size=(9, 9),    # 在模型中加入第二个卷积层，（之前先产生32张，然后由32变64）创建64个卷积输入核，卷积核大小为9*9，
                 activation='relu',                 # 指定激活函数（只有第一层需要指定输入形状）
                 padding='same'))                   # 表示处理后的图像与原图像大小一致
model.add(Dropout(0.25))                            # 加入第二个正则化层
model.add(MaxPooling2D(pool_size=(4, 4)))           # 加入第二个池化层
model.add(Flatten())                                # 加入平坦层，将前面的数据展开成一位
model.add(Dropout(rate=0.25))                       # 加入第三个正则化层
model.add(Dense(1024, activation='relu'))           # 加入隐藏层，对前面提取的特征进行第一步处理
model.add(Dropout(rate=0.25))                       # 继续加入正则化层
model.add(Dense(10, activation='softmax'))          # 创建输出层，因为我们的数据有十个类别，所以这里输入区，softmax时常用的多分类函数
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])  # 指定交叉熵作为损失函数，优化函数选择Adam，评价函数为准确李，即分类正确的占总样本的数量
train_history= model.fit(x_train, y_train,
                         validation_split=0.2,
                         epochs=1, batch_size=8, verbose=1)       # 读取训练集进行训练，验证集比例为0.2，训练20个周期，batch大小为8，显示方式为1

scores = model.evaluate(x_test, y_test, verbose=0)  # 用训练好的模型对测试集进行预测
scores[1]  # 查看预测结果
#
# # Save_Model('D:/Python 文件/Huawei/keras_cnn.py')
#
# os.chdir("D:/Python 文件/Huawei")
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# joblib.dump(clf, "train_model.m")
#
# new_clf = joblib.load("train_model.m")
# new_clf.evaluate(x_test)
# # clf = svm.SVC()
# # iris = datasets.load_iris()
# # X, y = iris.data, iris.target
# # clf.fit(X, y)
# # # save model
# # s = pickle.dumps(clf)
# # f = open('svm.model', 'w')
# # f.write(s)
# # f.close()



