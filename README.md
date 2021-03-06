# RecognizeCode
验证码识别，使用libsvm建立学习模型，进行识别
## 依赖
~~~python
pip install pillow
~~~
## 安装LibSVM
1. 从LIBSVM官网上https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download 下载所需要的安装LIBSVM版本，解压后文件夹重命名为libsvm，将之放入~Python36\Lib\site-packages目录，将libsvm文件夹下的tools和windows这两个文件夹加入到系统路径path里;
2. 到https://www.lfd.uci.edu/%7Egohlke/pythonlibs/#libsvm 下载对应版本的libsvm，pip进行安装成功后，在~Python36\Lib\site-packages目录下找到新生成的libsvm.dll，将其放到C:\windows\system32
## LibSVM训练并识别
1. 将爬取的验证码图片保存到codes目录下
2. 先只运行training.py中croptest()函数，将会把验证码图片切割为单个图片，并自动保存到Training目录，然后手动分类，一类一个文件夹
3. 分类完成后将croptest()函数注释，然后运行training.py脚本（注意路径），将自动在Training目录下生成特征文件feature.txt及训练文件model.txt
4. 运行GetCode.py，识别成功！
5. 如需识别其他类型的验证码图片，则需另外训练，此处只提供一种识别方案
## GetCodePic.py说明
1. 用于爬取验证码图片，自动保存到codes目录;
2. 使用urllib爬取，获取content_type字段判断图片后缀名
## training.py说明
1. 用于建立LibSVM训练库，生成训练模型文件；
2. croptest()函数用于将codes目录下的验证码图片切割为只含有单个元素的图片，并保存到Training目录，之后需要手动分类，分类好的效果如Training下的0~9文件夹；
3. GetFeatureFile()函数用于生成带特征值和标记值的libSVM向量文件，即Training目录下的feature.txt；在遍历目录时使用islice迭代，并跳过第一个元素
4. Train_SVM_model()函数则用于生成训练模型文件。
5. 使用时注意，先运行croptest()函数，将生成的训练图片分类好后再运行下面的内容
## GetCode.py说明
1. 主函数GetCode()先将图片转化为灰度图，再将图片二值化处理，再将图片使用de_nise()函数降噪，然后切割成只有单个元素的图片，最后用训练好的模型进行识别；
2. de_noise()降噪函数使用简易洪水填充算法，判断一个像素点的四周有多少个像素点与之相同，如果小于某个数，则将该像素点置为白色（或黑色）
3. GetFeature() 函数用于从切割后的图片中提取特征，此处采用提取每一行每一列像素值为0（黑色）的个数作为该图片的特征，行列即为该图片的size，如图片size为11\*17，则行为11，列为17，记录11行中每一行中像素为0的像素点的个数，对于列也同样如此，二者相加，共28组数据；也可以采用相乘的方法作为特征。
4. 运行该脚本，可成功识别已训练的那一种类型的验证码图片。
