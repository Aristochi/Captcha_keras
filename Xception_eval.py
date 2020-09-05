import os

from keras.models import load_model
import numpy as np
from scipy import misc
from keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import glob
import pandas as pd
img_size = (40, 120)
model = load_model('./CaptchaXception30.h5')
# model = load_model('./xceptbest.h5')
letter_list = [chr(i) for i in range(48, 58)] +[chr(i) for i in range(65, 91)]+[chr(i) for i in range(97, 123)]


def data_generator_test(data, n):  # 样本生成器，节省内存
    while True:
        batch = np.array([data[n]])
        x, y = [], []
        for img in batch:
            im = misc.imread(img)
            im = im[:, :, :3]
            im = misc.imresize(im, img_size)
            x.append(im)  # 读取resize图片,再存进x列表
            y_list = []

            real_num=str(img).split('\\')[1].split('.')[0]
            for i in real_num:
                i = i.upper()
                if 48 <= ord(i) <= 57:
                    y_list.append(ord(i) - 48)
                elif 65 <= ord(i) <= 90:
                    y_list.append(ord(i) - 55)
                elif 97 <= ord(i) <= 122:
                    y_list.append(ord(i) - 61)
                else:
                    print(real_num)
            y.append(y_list)

            y.append(y_list)  # 把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
            # print('real_1:',img[-8:-4])
        x = preprocess_input(np.array(x).astype(float))  # 原先是dtype=uint8转成一个纯数字的array
        y = np.array(y)

        yield x, [y[:, i] for i in range(4)]


# def predict2(n):
#     x, y = next(data_generator_test(test_samples, n))
#     z = model.predict(x)
#     z = np.array([i.argmax(axis=1) for i in z]).T
#     result = z.tolist()
#     v = []
#     for i in range(len(result)):
#         for j in result[i]:
#             v.append(letter_list[j])
#
#     # 输出测试结果
#     str = ''
#     for i in v:
#         str += i
#
#     real = ''
#     for i in y:
#         for j in i:
#             real += letter_list[j]
#     return (str, real)
#

# test_samples = glob.glob(r'./val/*.jpg')
#
# n = 0
# n_right = 0
# for i in range(len(test_samples)):
#     n += 1
#     print('~~~~~~~~~~~~~%d~~~~~~~~~' % (n))
#     predict, real = predict2(i)
#
#     if real == predict:
#         n_right += 1
#     else:
#         print('real:', real)
#         print('predict:', predict)
#         image = mpimg.imread(test_samples[i])
#         plt.axis('off')
#         plt.imshow(image)
#         plt.show()
#
# print(n, n_right, n_right / n)
# test_samples1 = glob.glob(r'./dataset/test/*.jpg')
# test_list = [i for i in range(len(test_samples1))]
def data_generator_test1(data, n): #样本生成器，节省内存
    while True:
        batch = np.array([data[n]])
        x,y = [],[]
        for img in batch:
            x.append(misc.imresize(misc.imread(img), img_size)) #读取resize图片,再存进x列表
            y.append([ord(i)-ord('a') for i in img[-8:-4]]) #把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
        x = preprocess_input(np.array(x).astype(float)) #原先是dtype=uint8转成一个纯数字的array
        y = np.array(y)
        yield x,[y[:,i] for i in range(4)]


def predict1(img):
    x = []
    x.append(misc.imresize(misc.imread(img), img_size))
    x = preprocess_input(np.array(x).astype(float))
    z = model.predict(x)
    # print(z)
    z = np.array([i.argmax(axis=1) for i in z]).T
    # print(z)
    # print('----------------------')
    result = z.tolist()
    v = []
    for i in range(len(result)):
        for j in result[i]:
            v.append(letter_list[j])
    # image = mpimg.imread(test_samples1[n])
    # plt.axis('off')
    # plt.imshow(image)
    # plt.show()
    #输出测试结果
    str = ''
    for i in v:
        str += i
    return (str)
# print(predict1('./dataset/train/3.jpg'))
def pack(file_names, labels):
    frame = pd.DataFrame({'ID':file_names,'label':labels},columns=['ID','label'])
    return frame

imgpath='./test/'
imglist=os.listdir(imgpath)
imglist.sort(key=lambda x:int(x[:-4]))
predict_file_name=[]
labels=[]
for i in imglist:
    predict_file_name.append(str(i))
    # print(imgpath+str(i))
    label=predict1(imgpath+str(i))
    # print(label)
    labels.append(label)
frame = pack(predict_file_name, labels)
frame.to_csv('./04.csv', index=False)