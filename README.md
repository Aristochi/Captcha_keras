# Captcha_keras
Keras预训练cnn模型Xception迁移学习验证码识别（大小写字母+数字）
Keras预训练cnn模型Xception迁移学习验证码识别（大小写字母+数字），深度学习验证码识别62类字符（0~10）（a~z）(A~Z)，使用imgaug数据增强
详细博客文章可以看https://blog.csdn.net/Aristochi/article/details/106605109
数据集文件
链接：https://pan.baidu.com/s/1u4IP4cspT0_g7PnbEZw9mw
提取码：0qji
复制这段内容后打开百度网盘手机App，操作更方便哦（手动狗头doge.jpg）
另外我放的是我没有修改过的数据集，网盘里的图片不是按照xxxx.jpg这种格式的，
而是1.jpg这种按顺序的，不过有对应标签的csv文件，为了方便我就把图片用对应的label重命名了一下，
数据集读取的时候也可以用csv文件来读取数据，
id是图片名称，label是对应标签，
文件路径+id的方式读取图片和对应图片的label

后续继续训练以及数据增强训练以后单个模型的准确率是86.733%，
从比赛角度上可以生成预测的多个结果csv文件，
通过集成学习（Voting投票）获得最佳的csv文件，或许可以达到90%以上的准确率。
此外，可以读数据集做图像处理，例如高斯模糊去掉干扰线，灰度图二值化等去掉干扰背景。
