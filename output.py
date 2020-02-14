import cv2
import numpy as np
from skimage import transform
import keras
import os
import pickle
import tensorflow as tf
from sklearn import datasets

from keras.backend.tensorflow_backend import set_session

from sklearn.externals import joblib
# iris = datasets.load_iris()
# X, y = iris.data, iris.target
# f2 = open('svm.model','r')
# s2 = f2.read()
# clf2 = pickle.loads(s2)
# clf2.predit(X, y)

# def Load_Model(self, filepath):
#     model = joblib.load(filepath)
#     return model


# g = tf.Graph()
# with g.as_default():
#     pass

# c = tf.constant(4.0)
# assert c.graph is tf.compat.v1.get_default_graph()  # 看看主程序中新建的一个变量是不是在默认图里
# g = tf.Graph()
# with g.as_default():
#     c = tf.constant(30.0)


# # 清理开始会话的session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)
# tf.compat.v1.keras.backend.clear_session()  # 此句代码分量很重

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
w = 128
h = 128
CKPT_DIR = 'ckpt'

#
# def read_data(path):
#     image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     print(path)
#     image = transform.resize(image, output_shape=(w, h))  # 转换图像的大小
#     return image
#

# def predict(image_path, model, output):
#     image = read_data(image_path)
#     result = model.run(output)
#     result = np.argmax(result, 1)
#     print('The prediction is', result)
#     cv2.addText(image, 'The prediction is {}'.format(result), (20, 20), cv2.FONT_HERSHEY_SIMPLEX)
#     cv2.waitkey(0)
#     cv2.destroyAllWindows()


# class Predict:
    # def __init__(self):
    #     with tf.compat.v1.Session() as ses:
    #         self.ses = tf.compat.v1.Session()
    #     self.ses.run(tf.compat.v1.global_variables_initializer())
    #     self.restore()  # 加载模型到sess中
    #
    # def restore(self):
    #     saver = tf.compat.v1.train.Saver()
    #     ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(self.sess, ckpt.model_checkpoint_path)
    #     else:
    #         raise FileNotFoundError("未保存任何模型")
    #
    # def predict(self, image_path):
    #     with tf.compat.v1.Session() as ses:
    #         self.ses = tf.compat.v1.Session()
    #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #
    #         x = np.array([1])
    #         y = self.ses.run(self.net.y, feed_dict={self.net.x: x})

        # 因为x只传入了一张图片，取y[0]即可
        # np.argmax()取得独热编码最大值的下标，即代表的数字
        print(image_path)
        print('        -> Predict digit', np.argmax(y[0]))


new_clf = joblib.load("train_model.m")
app = Predict()
# new_model = keras.models.load_model('my_model.h5')
# Load_Model()
# new_model.summary()
app.predict('C:/Users/PZEZ/Desktop/0.jpg')
new_clf.predit(test_X)  #此处test_X为特征集


