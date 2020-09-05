import numpy as np
import glob
from keras.applications.xception import Xception, preprocess_input
from keras.engine.saving import load_model
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from scipy import misc
samples = glob.glob('./train/*.jpg')  # 获取所有样本图片
np.random.shuffle(samples)  # 将图片打乱
nb_train = 18000  # 共有2万样本，1.8万用于训练，2k用于验证
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]
img_size = (40, 120)
input_image = Input(shape=(img_size[0], img_size[1], 3))
# 直接将验证码输入，做几个卷积层提取特征，然后把这些提出来的特征连接62个分类器，
# 输入图片
# 用预训练的Xception提取特征,采用平均池化
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')
# 用全连接层把图片特征接上softmax然后62分类，dropout为0.5
# Softmax - 用于多分类神经网络输出
predicts = [Dense(62, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model = load_model('./CaptchaXception.h5')#可以加载模型继续训练
def data_generator(data, batch_size):  # 样本生成器，节省内存
    while True:
        # 生成一个从x中抽取的随机数,维度为y的向量，y为抽取次数
        batch = np.random.choice(data, batch_size)
        x, y = [], []
        for img in batch:
            x.append(misc.imresize(misc.imread(img), img_size))  # 读取resize图片,再存进x列表
            imglabel=str(img).split('\\')[1].split('.')[0]
            label=str(imglabel)
            # print(label)
            y_list = []
            for i in label:
                # i = i.upper()
                if 48 <= ord(i) <= 57:
                    y_list.append(ord(i) - 48)
                if 65 <= ord(i) <= 90:
                    y_list.append(ord(i) - 55)
                if 97 <= ord(i) <= 122:
                    y_list.append(ord(i) - 61)
            y.append(y_list)
        # 把验证码标签添加到y列表,把对应字母转化为数字
        x = preprocess_input(np.array(x).astype(float))
        # 原先是dtype=uint8转成一个纯数字的array
        y = np.array(y)
        yield x, [y[:, i] for i in range(4)]
        # 输出：图片array和四个转化成数字的字母 例如：[array([6]), array([0]), array([3]), array([24])])
ep=10
while True:
    model.fit_generator(data_generator(train_samples, 48), steps_per_epoch=375, epochs=5,
                    validation_data=data_generator(test_samples, 48), validation_steps=40)
    model.save('CaptchaXception'+str(ep)+'.h5')
    ep+=10
# 保存模型
