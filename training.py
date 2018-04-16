import os
from itertools import islice
from PIL import Image
from libsvm.python.svm import *
from libsvm.python.svmutil import *
from GetCode import GetFeature,GetFeatureStr,de_noise,CropImgs

def croptest(dpath):#将目录下的验证码图片切割并保存到Training目录,即创建训练图片，之后需要手动分类
    pics=list(os.walk(dpath))[0][2]
    trainpath=os.path.join(os.path.split(dpath)[0],'Training')
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        suffix=os.path.splitext(pic)[1]
        im=Image.open(os.path.join(dpath,pic)).convert('L')#打开图片，并转化为灰度图
        im=im.point([0]*150+[1]*(256-150),'1')#二值化
        im=de_noise(im)#降噪
        Imgs=CropImgs(im)#切割后的图片列表
        for img in Imgs:
            n+=1
            img.save(os.path.join(trainpath,'%d%s'%(n,suffix)))
    return trainpath
#croptest(r'.\codes')

def GetFeatureFile(Fpath):#生成带特征值和标记值的libSVM向量文件,feature.txt
    #Fpath为原子图片所在目录的父目录，即Training目录，需要将里面的图片分类
    feature=os.path.join(Fpath,'feature.txt')
    for path,files,pics in islice(os.walk(Fpath),1,None):#使用islice迭代，跳过第一个元素
        for pic in pics:
            with open(feature,'a') as f:
                f.writelines(GetFeatureStr(GetFeature(os.path.join(path,pic),'path'),os.path.split(path)[1]))
    return feature#返回生成的向量文件路径

def Train_SVM_model(PathToFeatureFile):#生成训练模型文件，model.txt
    y,x=svm_read_problem(PathToFeatureFile)
    model=svm_train(y,x)
    modelpath=os.path.join(os.path.split(PathToFeatureFile)[0],'model.txt')
    svm_save_model(modelpath,model)   

if __name__ == '__main__':
    croptest(r'.\codes')#先运行这行，把下面两行注释；再把这行注释，运行下面两行
    #featurePath=GetFeatureFile(r'.\Training')
    #Train_SVM_model(featurePath)
