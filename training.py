import os,sys,svmutil
from itertools import islice
from PIL import Image
from GetCode import *

def crop_login(dpath):#将目录下的验证码图片切割并保存到Training目录,即创建训练图片，之后需要手动分类
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(dpath)[0],'Training')
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Image.open(os.path.join(dpath,pic)).convert('L')#打开图片，并转化为灰度图
        im=im.point([0]*150+[1]*(256-150),'1')#二值化
        im=de_noise(im)#降噪
        Imgs=Crop_login(im)#切割后的图片列表
        for oldimg in Imgs:
            n+=1
            img=createNewimage(oldimg)
            img.save(os.path.join(trainpath,'%d%s'%(n,'.png')))
    return trainpath
#crop_login(r'.\codes')

def crop_register(dpath):#注册验证码
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(dpath)[0],'Training2')
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Pre_process_register(os.path.join(dpath,pic)) 
        try:       
            Imgs=CorpForRegister(im)#切割后的图片列表
        except Exception as e:
            print(e)
            print(os.path.join(dpath,pic))
            sys.exit(0)
        for img in Imgs:
            n+=1
            #img=createNewimage(oldimg)
            img.save(os.path.join(trainpath,'%d_%s'%(n,pic)))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath

def crop_bigboss(dpath,spath):#大总管验证码
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(os.path.split(dpath)[0])[0],spath)
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Pre_process_bigboss(os.path.join(dpath,pic))     
        Imgs=Crop_bigboss(im)#切割后的图片列表
        for img in Imgs:
            n+=1
            #img=createNewimage(oldimg)
            img.save(os.path.join(trainpath,'%d_%s.png'%(n,os.path.splitext(pic)[0])))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath

def crop_OM(dpath,spath):#运营管理验证码
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(os.path.split(dpath)[0])[0],spath)
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Pre_process_OM(os.path.join(dpath,pic))       
        Imgs=Crop_OM(im)#切割后的图片列表
        if len(Imgs)!=4:print('%s 分割失败'%pic)
        for img in Imgs:
            n+=1
            #img=createNewimage(oldimg)
            img.save(os.path.join(trainpath,'%d_%s.png'%(n,os.path.splitext(pic)[0])))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath

def crop_MM(dpath,spath):#资金管理验证码
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(os.path.split(dpath)[0])[0],spath)
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Pre_process_MM(os.path.join(dpath,pic))       
        Imgs=Crop_MM(im)#切割后的图片列表
        for img in Imgs:
            n+=1
            img=upright(img)
            img.save(os.path.join(trainpath,'%d_%s.png'%(n,os.path.splitext(pic)[0])))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath    

def crop_MC(dpath):#消息中心验证码
    #dpath=r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_MM1'
    #trainpath='E:\\Users\\ZP\\Desktop\\5-2\\py\\RecognizeCode\\Trainings\\Training_MM1'
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(os.path.split(dpath)[0])[0],f"Trainings\\Training_{dpath.split('_')[1]}")
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        #suffix=os.path.splitext(pic)[1]
        im=Pre_process_MC(os.path.join(dpath,pic))
        Imgs=Crop_Vertical(im,th=1,maxlenth=24)#切割后的图片列表
        if len(Imgs)!=4:print('错误：',pic,' '*10)
        for img in Imgs:
            n+=1
            img.save(os.path.join(trainpath,'%d_%s.png'%(n,os.path.splitext(pic)[0])))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath

def crop_H5(dpath):#消息中心验证码
    #dpath=r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_MM1'
    #trainpath='E:\\Users\\ZP\\Desktop\\5-2\\py\\RecognizeCode\\Trainings\\Training_MM1'
    pics=list(os.walk(dpath))[0][2]#获取目录下的所有文件名
    trainpath=os.path.join(os.path.split(os.path.split(dpath)[0])[0],f"Trainings\\Training_{dpath.split('_')[1]}")
    if not os.path.exists(trainpath):os.mkdir(trainpath)
    n=0
    for pic in pics:
        im=de_noise(Image.open(os.path.join(dpath,pic)).convert('L').point([0]*160+[1]*(256-160),'1'))
        try:
            Imgs=Crop_Vertical(im,th=3,maxlenth=30)#切割后的图片列表
        except UnboundLocalError:
            print('UnboundLocalError错误：',pic,' '*10)
        if len(Imgs)!=4:print('错误：',pic,' '*10)
        for img in Imgs:
            n+=1
            img.save(os.path.join(trainpath,'%d_%s.png'%(n,os.path.splitext(pic)[0])))
            sys.stdout.write('%d %s       \r'%(n,pic))
            sys.stdout.flush()
    return trainpath
#############################################################################################
def GetFeatureFile(trainingPath):#生成带特征值和标记值的libSVM向量文件,feature.txt
    #trainingPath为原子图片所在目录的父目录，需要将里面的图片分类
    modelpath=os.path.join(trainingPath.split('Trainings')[0],'model')
    if not os.path.exists(modelpath):os.mkdir(modelpath)
    feature=os.path.join(modelpath,f"feature_{trainingPath.split('_',1)[1]}.txt")
    for path,files,pics in islice(os.walk(trainingPath),1,None):#使用islice迭代，跳过第一个元素
        for pic in pics:
            with open(feature,'a') as f:
                f.writelines(GetFeatureStr(GetFeature(os.path.join(path,pic),'path'),ord(os.path.split(path)[1]) ))
            sys.stdout.write('%s %s    \r'%(os.path.split(path)[1],pic))
            sys.stdout.flush()
    return feature#返回生成的向量文件路径

def Train_SVM_model(PathToFeatureFile):#生成训练模型文件，model.txt
    #print(PathToFeatureFile)
    y,x=svmutil.svm_read_problem(PathToFeatureFile)
    model=svmutil.svm_train(y,x)
    modelFilePath=os.path.join(os.path.split(PathToFeatureFile)[0],f"model_{PathToFeatureFile.split('_',1)[1]}")
    svmutil.svm_save_model(modelFilePath,model)
    print(modelFilePath)

if __name__ == '__main__':
    # crop(r'.\codes3\failed')#
    # crop2(r'.\codes5\failed')
    # crop_bigboss(r'.\codes\codes_bigboss2',r'Trainings\Training_bigboss1')
    # crop_OM(r'.\codes\codes_OM1',r'Trainings\Training_OM')
    # crop_MM(r'.\codes\codes_MM',r'Trainings\Training_MM')
    # crop_MC(r'.\codes\codes_MC1')
    # crop_H5(r'.\codes\codes_H5')
    # GetFeatureFile(r'.\Training1')
    # Train_SVM_model('./model1/feature.txt')
    # Train_SVM_model(GetFeatureFile(r'Trainings\Training_H5'))
    # Train_SVM_model(GetFeatureFile(r'Trainings\Training_SKB'))
    # Train_SVM_model(GetFeatureFile(r'Trainings\Training_SKB_abc'))
    Train_SVM_model(GetFeatureFile(r'Trainings\Training_SKB_123'))
