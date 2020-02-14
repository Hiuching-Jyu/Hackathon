from sklearn.externals import joblib
from keras.models import Sequential  # keras里面用于创建深度学习模型的包
from keras_cnn import x_test, y_test, model

scores = model.evaluate(x_test, y_test, verbose=0)  # 用训练好的模型对测试集进行预测
scores[1]  # 查看预测结果

new_clf = joblib.load("train_model.m")
new_clf.evaluate(x_test)