import os
import tensorflow as tf
import numpy as np               # 是python中一个高效的数值计算库，而图像在解码之后都是矩阵数据，需要用到
import matplotlib.pyplot as plt


# 定义图像预处理，随即调整图像的色彩，因为调整图像参数的顺序会影响到最后结果，所以预处理时可设置为随机使用不同顺序，降低无关因素对模型的影响
def random_color(image, color_order=0):                                  # 首先给调整图像参数的顺序设置为=0
    if color_order == 0:                                                 # 随机选择一种调整顺序
        image = tf.image.random_brightness(image, max_delta=32./255.)    # 随即调整亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 随机调整饱和度
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)    # 随即调整对比度
    elif color_order==1:                                                 # 随机改变调整图片参数的顺序，这里只采用了两种
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image, 0.0, 1.0)                             # 随机给图像赋值1 or 0，该函数第二三个参数分别为最小值&最大值


# 给定 一张解码后的图像、目标图像尺寸以及图像上的注释框，此函数可以对给出的图像进行预处理。这个函数的输入图像是
# 图像识别问题中的原始的训练图像，而输出则是神经网络模型的输入层。注意这里只处理模型的训练数据，对于预测的数据，
# 一般不需要使用随机变换的步骤
def preprocess_for_train(image, height, width, bbox):
    if bbox is None:                                        # 判断是否有标注框，没有的话则将整个图像作为需要关注的部分
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                          dtype=tf.float32, shape=[1, 1, 4])       # 生成默认的标注框，shape的三个参数分别代表batch(处理一组图片），与图像相关联的N个边界框的形状，4个参数
    if image.dtype != tf.float32:                           # 检测图像张量的类型
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)                     # 把数据类型更改为浮点型，类型统一便于高效运算
    bbox_begin, bbox_size, _=tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    random_image = tf.slice(image, bbox_begin, bbox_size)  # 用bbox解析得到的标注框对图像进行裁剪，得到新图像
    # 将截取的图像调整为神经网络输入层的大小，大小调整的算法是随机选择的，tensorflow中提供了四种方法，
    # 所以这里用randint（4）在四种方法中随机选择一种
    random_image = tf.image.resize_images(random_image, [height, width], method=np.random.randint(4))   # np.random.randint是一个返回一个随机数或者随机数组的函数，此处4 代表开区间最大值，但为什么不是5呢？
    random_image = tf.image.random_flip_left_right(random_image)    # 随机左右翻
    # 转图像
    random_image = random_color(random_image, np.random.randint(2)) # 使用一种随机顺序调整色彩
    return random_image                                           # 返回最终调整好的图像


image_raw_data = tf.io.gfile.GFile('img-classified/0/0.jpg', 'rb').read()    # 利用tensorflow自带的函数读取图片,原来此处是FastGFile，出现报错
with tf.Session() as sess:                                                          # 对上一部分读取的图像内容进行解码
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])        # 设置标注框，需注意：标注框是tensorflow中的一个常量
    for i in range(6):                                                              # 运行六次，得到六张图像
        result = preprocess_for_train(img_data, 600, 600, boxes)                    # 利用前面的预处理对图像进行处理，并得到结果
        plt.imshow(result.eval())                                                   # 绘制预处理之后的图像
        plt.show()                                                                  # 显示图像
