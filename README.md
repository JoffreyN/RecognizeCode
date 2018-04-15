# RecognizeCode
验证码识别，使用libsvm建立学习模型，进行识别
## 依赖
~~~python
pip install pillow
~~~
## 安装LibSVM
1. 从LIBSVM官网上https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download 下载所需要的安装LIBSVM版本，解压后文件夹重命名为libsvm，将之放入~Python36\Lib\site-packages目录，将libsvm文件夹下的tools和windows这两个文件夹加入到系统路径path里;
2. 到https://www.lfd.uci.edu/%7Egohlke/pythonlibs/#libsvm 下载对应版本的libsvm，pip进行安装成功后，在~Python36\Lib\site-packages目录下找到新生成的libsvm.dll，将其放到C:\windows\system32
## LibSVM训练
1. 将爬取的验证码图片保存到codes目录下
2. 先只运行training.py中croptest()函数，将会把验证码图片切割为单个图片，并自动保存到Training目录，然后手动分类，一类一个文件夹
3. 分类完成后将croptest()函数注释，然后运行training.py脚本（注意路径），将自动在Training目录下生成特征文件feature.txt及训练文件model.txt
4. 修改GetCode.py里需要识别的图片路径后运行，成功
5. 如需识别其他类型的验证码图片，则需另外训练，此处只提供一种识别方案
