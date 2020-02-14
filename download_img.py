import pandas as pd      # pandas是一个可以处理csv，xls等格式文件的包
import requests          # requestes 包可以让python获得网上的数据，本实验用于获取网上的图片
import os                # os负责系统和python之间的通信，用于创建新文件夹

content = pd.read_csv('D:/Python 文件/基于卷积神经网络实现景区精准识别实验指导V2.5/train.csv')
# 此处理解为：pandasz读取train.csv中的数据，并传送到content中
content.describe()  # 此处理解为：在运行输出中，查看数据集的描述，包括几个参数

arr = ['0', '2', '3', '5', '7', '8', '11', '12', '13', '16']  # 自己创建一个列表，此处的数值分别对应image文件里边的10个文件夹，分别装有不同的数据
content_selected = content[content['landmark_id'].isin(arr)]  # 用isin(list)函数，引入并检查arr这个列表中的元素是否都在“landmark_
# id”这个列表中
content_selected = content_selected.reset_index()  # 此处不是很理解：筛选后的数据及还是保持了原来的索引，且不连续，因此需要重置
content_selected.head()                            # emm据说是查看数据集的前面部分，默认前5项
num = content_selected.shape[0]                    # content_selected.shape 中储存的元素是一个表格


# shape[0]代表的是表格的行数，shape[1]代表的是表格的列数


def download_img(num):
    for i in range(num):                                # range()的用法：range(起始值，结束值） or range(结束值）--->默认从0开始计
        img_url = content_selected.at[i, 'url']         # at的用法Get value at specified row/column pair，i表示指定行，‘url’表示指定列
        img_name = str(i) + '.jpg'                      # 用循环序号作为图片的名字，需要用str()把数值转换为字符串
        folder = content_selected.at[i, 'landmark_id']  # 创建一个文件夹，并以景点的名字命名
        path = 'img/' + folder + '/'                    # 定义保存图片的路径，在img文件夹中创建该以景点名字命名的文件夹
        isexists = os.path.exists(path)                 # 用于判断路径是否存在，如果文件夹还没有创建，文件不存在的话，写入文件会报错
        if not isexists:
            os.makedirs(path)                           # 这部分利用os模块创建文件夹
        # try 和except负责python里边的异常捕捉和处理
        try:
            img = requests.get(img_url)             # 尝试打开、访问前面拿到的图片链接
            with open(path + img_name, 'ab') as f:  # with open as ...with关键字可以在这部分内容万恒后，自动关闭刚才打开的资源，不用手动管理
                # open则是表示在前面设置的路径里边新建、打开一个文件夹
                f.write(img.content)                # write 将访问图片链接拿到的内容写入到空白文件中，这部分完成后，with 会自动保存内容并释放资源
        except OSError:                             # 此处对比原文稍作了改动，因为有报错显示这个except的范围太大了需要缩小范围，于是随便选了一个报错上去，反正最后也pass掉了
            pass
