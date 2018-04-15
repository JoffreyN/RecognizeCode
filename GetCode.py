import os,re
from PIL import Image
from libsvm.python.svm import *
from libsvm.python.svmutil import *

def GetCode(picpath):#使用libsvm训练模型
    code=[]
    im=Image.open(picpath).convert('L')#打开图片，并转化为灰度图
    im=im.point([0]*150+[1]*(256-150),'1')#二值化
    #im.show()
    im=de_noise(im)#降噪
    #im.show()
    Imgs=CropImgs(im)#切割后的图片列表
    model=svm_load_model(r'.\Training\model.txt')#打开训练文件
    for img in Imgs:
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svm_read_problem(tempath)        
        p_label,p_acc,p_val=svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(str,code)))
    return code

def de_noise(im):#降噪，简易洪水填充算法
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if x==0:#最左边列
                if y==0:#左上顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)))<2:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#左下顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x,y-1)),im.getpixel((x+1,y-1)))<2:
                        im.putpixel((x,y),1)
                else:#最左边列非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x+1,y-1)),im.getpixel((x+1,y)),im.getpixel((x+1,y+1)),im.getpixel((x,y+1)))<2:
                        im.putpixel((x,y),1)
            elif x==im.width-1:#最右边列
                if y==0:#右上顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)))<2:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#右下顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)))<2:
                        im.putpixel((x,y),1)
                else:#最右边列非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)))<2:
                        im.putpixel((x,y),1)
            else:#中间的列
                if y==0:#最上边非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)),im.getpixel((x+1,y)))<2:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#最下边非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x+1,y-1)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)))<2:
                        im.putpixel((x,y),1)
                else:#中间的点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)),im.getpixel((x+1,y)),im.getpixel((x+1,y-1)),im.getpixel((x,y-1)))<2:
                        im.putpixel((x,y),1)
    return im
                    
def Equal_Points(center,*arounds):#返回与中心点像素相同的四周的点的个数
    number=0
    for i in arounds:
        if center==i:number+=1
    return number

def CropImgs(img):#切割图片
    child_img_list=[]
    for i in range(4):
        x=6+i*20
        y=9
        child_img=de_noise(img.crop((x,y,x+11,y+17)))
        #child_img.show()
        child_img_list.append(child_img)
    return child_img_list

def GetFeature(imgORpicpath,model):#从切割后的图片中获取特征
    if model=='path':img=Image.open(imgORpicpath)
    elif model=='img':img=imgORpicpath
    width,height=img.size
    pixel_cnt_list=[]
    for y in range(height):
        pix_cnt_x=0
        for x in range(width):
            if img.getpixel((x,y))==0:pix_cnt_x+=1
        pixel_cnt_list.append(pix_cnt_x)
    for x in range(width):
        pix_cnt_y=0
        for y in range(height):
            if img.getpixel((x,y))==0:pix_cnt_y+=1
        pixel_cnt_list.append(pix_cnt_y)
    return pixel_cnt_list

def GetFeatureStr(FeatureList,*Feature):#将整型特征列表转换为字符型列表
    if Feature:FeatureStr=['%s '%str(Feature[0])]
    else:FeatureStr=[]
    for i in range(len(FeatureList)):
        FeatureStr.append('%d:%d '%(i+1,FeatureList[i]))
    if Feature:FeatureStr.append('\n')
    else:FeatureStr=''.join(FeatureStr)
    return FeatureStr

def test(dpath):#识别目录下的验证码图片，并将识别结果作为图片名字
    pics=list(os.walk(dpath))[0][2]
    path=list(os.walk(dpath))[0][0]
    codelist=[]
    n=0
    for pic in pics:
        code=re.sub(r'\\|/|:|\*|\?|"|<|>|\|','_',CheckCode2(os.path.join(path,pic)))
        if code in codelist:
            n+=1
            code='%s-%d'%(code,n)
        newname=code+'.png'
        os.rename(os.path.join(path,pic),os.path.join(path,newname))
        codelist.append(code)
#test('E:\\Zp\\Desktop\\codes')
if __name__ == '__main__':
    print(GetCode(r'.\codes\15.png'))
