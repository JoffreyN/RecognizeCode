import os,re,uuid,sys,math,svmutil
from PIL import Image
from RecognizeCode import GetCode
#from DropWater import Vertical,Count_0,Appetizer
#key={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'a',11:'b',12:'c',13:'d',14:'e',15:'f',16:'g',17:'h',18:'i',19:'j',20:'k',21:'l',22:'m',23:'n',24:'o',25:'p',26:'q',27:'r',28:'s',29:'t',30:'u',31:'v',32:'w',33:'x',34:'y',35:'z'}
modelpath_login_register=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_login_register.txt')
modelpath_bigboss=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_bigboss.txt')
modelpath_OM=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_OM.txt')
modelpath_MM=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_MM.txt')
modelpath_MC=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_MC.txt')
modelpath_SKB=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_SKB.txt')
modelpath_SKB_abc=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_SKB_abc.txt')
modelpath_SKB_123=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_SKB_123.txt')
modelpath_H5=os.path.join(os.path.split(GetCode.__file__)[0],r'model\model_H5.txt')

#####登陆验证码#####start##########登陆验证码#####start##########登陆验证码#####start##########登陆验证码#####start##########登陆验证码#####start#####
def GetCode_login(picpath):#识别登陆验证码
    code=[];model=svmutil.svm_load_model(modelpath_login_register)#打开训练文件
    im=Image.open(picpath).convert('L')#打开图片，并转化为灰度图
    im=im.point([0]*150+[1]*(256-150),'1')#二值化
    im=de_noise(im)#降噪
    Imgs=Crop_login(im)#切割后的图片列表
    for oldimg in Imgs:
        img=createNewimage(oldimg,oldimg.size)
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code

def createNewimage(oldimage,size=(28,31)):
    newimage=Image.new('L',size,255)
    newimage=newimage.point([0]*150+[1]*(256-150),'1')
    for x in range(oldimage.size[0]):
        for y in range(oldimage.size[1]):
            newimage.putpixel((x,y),oldimage.getpixel((x,y)))
    return newimage

def Crop_login(img):#登陆验证码切割方法
    black_point_num=Vertical(img)
    x=[Count_0(black_point_num),30,53,74,108-Count_0(black_point_num[::-1])]
    child_img_list=[]
    for i in range(4):
        child_img=de_noise(img.crop((x[i],8,x[i+1],39)))
        child_img_list.append(child_img)
    return child_img_list
#####登陆验证码#####end############登陆验证码#####end############登陆验证码#####end############登陆验证码#####end############登陆验证码#####end#######

#####注册验证码#####start##########注册验证码#####start##########注册验证码#####start##########注册验证码#####start##########注册验证码#####start#####
def GetCode_register(picpath):#识别注册验证码
    code=[];model=svmutil.svm_load_model(modelpath_login_register)#打开训练文件
    im=Pre_process_register(picpath) 
    Imgs=Crop_Vertical(im,th=3)#切割后的图片列表
    for img in Imgs:
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code  

def Pre_process_register(picpath):#注册验证码前期处理，去除干扰线，灰度图，二值化
    im=Image.open(picpath)
    w,h=im.size
    for x in range(w):
        for y in range(h):
            if max(im.getpixel((x,y)))<100:
                im.putpixel((x,y),Effective_pixels(im,x,y))
    im=im.convert('L')
    im=im.point([0]*150+[1]*(256-150),'1')
    im=de_noise(im)
    return im

def Effective_pixels(img,x,y):
    #取中心点周围16邻域像素点R,G,B，分别计算白点和字符点个数，若白点比字符点多，则返回白点R,G,B，否则返回字符点R,G,B中差值最大的那个
    w,h=img.size
    white_point,Effective_point,pixel=0,0,[]
    for coordinate in [(x-2,y-2),(x-1,y-2),(x,y-2),(x+1,y-2),(x+2,y-2),(x-2,y-1),(x-1,y-1),(x,y-1),(x+1,y-1),(x+2,y-1),(x-2,y),(x-1,y),(x,y),(x+1,y),(x+2,y),(x-2,y+1),(x-1,y+1),(x,y+1),(x+1,y+1),(x+2,y+1),(x-2,y+2),(x-1,y+2),(x,y+2),(x+1,y+2),(x+2,y+2)]:#16领域
    #for coordinate in [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]:
        if (0<=coordinate[0]<w) and (0<=coordinate[1]<h):#判断是否为有效点
            if img.getpixel(coordinate)[0]>=200 and img.getpixel(coordinate)[1]>=200 and img.getpixel(coordinate)[2]>=200:
                white_point+=1#计算白点个数
                continue
            if 100<=max(img.getpixel(coordinate))<220:
                Effective_point+=1#计算字符点个数
                pixel.append(img.getpixel(coordinate))
    if white_point>=Effective_point:return (255,255,255)#白点比字符点多，返回白点
    else:
        if len(pixel)==0:return (255,255,255)
        else:return pixel[list(map(lambda pix:max(pix)-min(pix),pixel)).index(max(list(map(lambda pix:max(pix)-min(pix),pixel))))]

#####注册验证码#####end############注册验证码#####end############注册验证码#####end############注册验证码#####end############注册验证码#####end#######

#####大总管验证码#####start########大总管验证码#####start########大总管验证码#####start########大总管验证码#####start########大总管验证码#####start####
def GetCode_bigboss(picpath):#识别大总管验证码
    code=[];model=svmutil.svm_load_model(modelpath_bigboss)#打开训练文件
    im=Pre_process_bigboss(picpath) 
    Imgs=Crop_bigboss(im)#切割后的图片列表
    for img in Imgs:
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code

def Pre_process_bigboss(picpath):#大总管验证码前期处理，去除干扰线，灰度图，二值化
    im=Image.open(picpath)
    w,h=im.size
    for x in range(w):
        for y in range(h):
            r,g,b=im.getpixel((x,y))
            if r-g>=90 and r>=150:
                im.putpixel((x,y),up_down_pixels(im,x,y))
    im=im.convert('L')
    im=im.point([0]*50+[1]*(256-50),'1')
    im=de_noise(im)
    return im

def up_down_pixels(img,x,y):
    w,h=img.size
    black_point=0
    for coor in [(x,y+1),(x,y-1)]:
        if 0<=coor[1]<h:
            if max(img.getpixel((coor)))<=40:
                   black_point+=1
    if black_point>0:return (0,0,0)
    else:return (255,255,255)

def Crop_bigboss(img):
    w,h=img.size
    child_img_list=[]
    for i in range(4):
        child_img=img.crop((10+16*i,0,10+16*(i+1),h))
        child_img_list.append(child_img)
    return child_img_list
#####大总管验证码#####end##########大总管验证码#####end##########大总管验证码#####end##########大总管验证码######end#########大总管验证码#####end######

#####运营验证码#####start##########运营验证码#####start##########运营验证码#####start##########运营验证码#####start##########运营验证码#####start#####
def GetCode_OM(picpath):#识别运营管理验证码
    code=[];model=svmutil.svm_load_model(modelpath_OM)#打开训练文件
    im=Pre_process_OM(picpath) 
    Imgs=Crop_OM(im)#切割后的图片列表
    for img in Imgs:
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code

def Pre_process_OM(picpath):#运营管理验证码前期处理，去除干扰线，灰度图，二值化
    im=Image.open(picpath)
    w,h=im.size
    im=im.convert('L')
    im=im.point([0]*210+[1]*(256-210),'1')
    im=de_noise(im)
    return im

def Crop_OM(img):#运营管理验证码切割方法
    w,h=img.size
    avg_w=round(w/4)
    child_img_list=[]
    vertical_point_num=Vertical(img)#垂直投影
    x_coordinate=Get_coordinate(vertical_point_num)
    for x in x_coordinate:
        child_img=img.crop((x[0],0,x[1]+1,h))#垂直分割
        level_point_num=Level(child_img)#水平投影
        child_img=child_img.crop((0,Count_0(level_point_num),child_img.size[0],child_img.size[1]-Count_0(level_point_num[::-1])))#水平分割
        if avg_w<=child_img.size[0]<2*avg_w:#垂直分割失败，暴力分割,连续两个字符
            child_img1=child_img.crop((0,0,child_img.size[0]/2,child_img.size[1]))
            child_img2=child_img.crop((child_img.size[0]/2,0,child_img.size[0],child_img.size[1]))
            for child in [child_img1,child_img2]:
                level_point_num=Level(child)
                child_img0=child.crop((0,Count_0(level_point_num),child.size[0],child.size[1]-Count_0(level_point_num[::-1])))
                child_img_list.append(child_img0)
        elif 2*avg_w<=child_img.size[0]<3*avg_w:#连续3个字符
            c_w=child_img.size[0]/3
            for i in range(3):
                child=child_img.crop((i*c_w,0,(i+1)*c_w,child_img.size[1]))
                child_img_list.append(child)
        else:
            child_img_list.append(child_img)
    return child_img_list
#####运营验证码#####end############运营验证码#####end############运营验证码#####end############运营验证码#####end############运营验证码#####end#######

#####资金验证码#####start##########资金验证码#####start##########资金验证码#####start##########资金验证码#####start##########资金验证码#####start#####
def GetCode_MM(picpath):#识别资金管理验证码
    code=[];model=svmutil.svm_load_model(modelpath_MM)#打开训练文件
    im=Pre_process_MM(picpath) 
    Imgs=Crop_MM(im)#切割后的图片列表
    for img in Imgs:
        w,h=img.size
        img=img.resize((w//5,h//5))
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code

def Crop_MM(img):#资金管理验证码切割方法
    w,h=img.size
    c_w=w/4
    child_img_list=[]
    for i in range(4):
        child_img=img.crop((i*c_w,0,(i+1)*c_w,h))
        child_img_list.append(child_img)
    return child_img_list

def Pre_process_MM(picpath,model='path'):#资金管理验证码前期处理，去除干扰线，灰度图，二值化
    if model=='path':
        img=de_noise(Image.open(picpath).convert('L').point([0]*180+[1]*(256-180),'1'))
    else:
        img=de_noise(picpath.convert('L').point([0]*180+[1]*(256-180),'1'))
    img=Expand(Corrode(img))#开运算.开运算是对图像先腐蚀后膨胀,闭运算是对图像先膨胀后腐蚀
    return img
#####资金验证码#####end############资金验证码#####end############资金验证码#####end############资金验证码#####end############资金验证码#####end#######

#####消息中心验证码#####start######消息中心验证码#####start######消息中心验证码#####start######消息中心验证码#####start######消息中心验证码#####start###
def GetCode_MC(picpath):#识别统一消息中心验证码
    code=[];model=svmutil.svm_load_model(modelpath_MC)#打开训练文件
    im=Pre_process_MC(picpath) 
    Imgs=Crop_Vertical(im,th=1,maxlenth=24)#切割后的图片列表
    for img in Imgs:
        w,h=img.size
        pixel_cnt_list=GetFeature(img,'img')        
        tempath=os.path.join(os.getcwd(),'temp.txt')#临时文件，用于存储将要识别的图片的特征
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)  
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code

def Pre_process_MC(picpath):
    img=Image.open(picpath)
    w,h=img.size
    for x in range(w):
        for y in range(h):
            r,g,b=img.getpixel((x,y))
            if 10<=r<=255 and 10<=g<=255 and 10<=b<=255:
                img.putpixel((x,y),(255,255,255))
            if x in [0,w-1] or y in [0,h-1]:
                img.putpixel((x,y),(255,255,255))
    img=de_noise(img.convert('L').point([0]*150+[1]*(256-150),'1'))
    return Fill_whitePoint(img)
#####消息中心验证码#####end########消息中心验证码#####end########消息中心验证码#####end########消息中心验证码#####end########消息中心验证码#####end#####

#######H5安全键盘#####start############H5安全键盘#####start############H5安全键盘#####start############H5安全键盘#####start############H5安全键盘#####
def GetCode_SKB(picpath,types='path'):
    model=svmutil.svm_load_model(modelpath_SKB)
    if types=='path':img=Image.open(picpath).convert('L').point([0]*165+[1]*(256-165),'1')
    elif types=='img':img=picpath
    pixel_cnt_list=GetFeature(img,'img')
    tempath=os.path.join(os.getcwd(),'temp.txt')
    with open(tempath,'w') as f:
        f.writelines(GetFeatureStr(pixel_cnt_list,0))
    y0,x0=svmutil.svm_read_problem(tempath)
    os.remove(tempath)
    p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
    return chr(int(p_label[0]))
#####H5安全键盘#####end########H5安全键盘#####end########H5安全键盘#####end########H5安全键盘#####end########H5安全键盘#####end#####H5安全键盘#########

#######H5手机版ABC键盘#####start############H5手机版ABC键盘#####start############H5手机版ABC键盘#####start############H5手机版ABC键盘#####start############
def GetCode_SKB_abc(picpath,types='path'):
    model=svmutil.svm_load_model(modelpath_SKB_abc)
    if types=='path':img=Image.open(picpath).convert('L').point([0]*165+[1]*(256-165),'1')
    elif types=='img':img=picpath
    pixel_cnt_list=GetFeature(img,'img')
    tempath=os.path.join(os.getcwd(),'temp.txt')
    with open(tempath,'w') as f:
        f.writelines(GetFeatureStr(pixel_cnt_list,0))
    y0,x0=svmutil.svm_read_problem(tempath)
    os.remove(tempath)
    p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
    return chr(int(p_label[0]))
#####H5手机版ABC键盘#####end########H5手机版ABC键盘#####end########H5手机版ABC键盘#####end########H5手机版ABC键盘#####end########H5手机版ABC键盘#####end#####

#######H5手机版纯数字键盘#####start############H5手机版纯数字键盘#####start############H5手机版纯数字键盘#####start############H5手机版纯数字键盘#####start#####
def GetCode_SKB_123(picpath,types='path'):
    model=svmutil.svm_load_model(modelpath_SKB_123)
    if types=='path':img=Image.open(picpath).convert('L').point([0]*165+[1]*(256-165),'1')
    elif types=='img':img=picpath
    pixel_cnt_list=GetFeature(img,'img')
    tempath=os.path.join(os.getcwd(),'temp.txt')
    with open(tempath,'w') as f:
        f.writelines(GetFeatureStr(pixel_cnt_list,0))
    y0,x0=svmutil.svm_read_problem(tempath)
    os.remove(tempath)
    p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
    return chr(int(p_label[0]))
#####H5手机版纯数字键盘#####end########H5手机版纯数字键盘#####end########H5手机版纯数字键盘#####end########H5手机版纯数字键盘#####end########H5手机版纯数字键盘####

#######H5验证码#######start############H5验证码#######start############H5验证码#######start############H5验证码#######start############H5验证码#######
def GetCode_H5(picpath,types='path'):#识别H5验证码
    code=[];model=svmutil.svm_load_model(modelpath_H5)#打开训练文件
    if types=='path':im=de_noise(Image.open(picpath).convert('L').point([0]*160+[1]*(256-160),'1'))
    elif types=='img':im=picpath
    Imgs=Crop_Vertical(im,th=3,maxlenth=30)
    for img in Imgs:
        w,h=img.size
        pixel_cnt_list=GetFeature(img,'img')
        tempath=os.path.join(os.getcwd(),'temp.txt')
        with open(tempath,'w') as f:
            f.writelines(GetFeatureStr(pixel_cnt_list,0))
        y0,x0=svmutil.svm_read_problem(tempath)
        os.remove(tempath)
        p_label,p_acc,p_val=svmutil.svm_predict(y0,x0,model,'-q')
        code.append(int(p_label[0]))
    code=''.join(list(map(chr,code)))
    return code
#####H5验证码#######end########H5验证码#######end########H5验证码#######end########H5验证码#######end########H5验证码#######end#####H5验证码###########

#######公共方法#####start############公共方法#####start############公共方法#####start############公共方法#####start############公共方法#####start####
def de_noise(im,degree=2):#降噪，简易洪水填充算法,0代表黑色
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if x==0:#最左边列
                if y==0:#左上顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)))<degree:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#左下顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x,y-1)),im.getpixel((x+1,y-1)))<degree:
                        im.putpixel((x,y),1)
                else:#最左边列非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x+1,y-1)),im.getpixel((x+1,y)),im.getpixel((x+1,y+1)),im.getpixel((x,y+1)))<degree:
                        im.putpixel((x,y),1)
            elif x==im.width-1:#最右边列
                if y==0:#右上顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)))<degree:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#右下顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)))<degree:
                        im.putpixel((x,y),1)
                else:#最右边列非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)))<degree:
                        im.putpixel((x,y),1)
            else:#中间的列
                if y==0:#最上边非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)),im.getpixel((x+1,y)))<degree:
                        im.putpixel((x,y),1)
                elif y==im.height-1:#最下边非顶点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x+1,y)),im.getpixel((x+1,y-1)),im.getpixel((x,y-1)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)))<degree:
                        im.putpixel((x,y),1)
                else:#中间的点
                    if Equal_Points(im.getpixel((x,y)),im.getpixel((x-1,y-1)),im.getpixel((x-1,y)),im.getpixel((x-1,y+1)),im.getpixel((x,y+1)),im.getpixel((x+1,y+1)),im.getpixel((x+1,y)),im.getpixel((x+1,y-1)),im.getpixel((x,y-1)))<degree:
                        im.putpixel((x,y),1)
    return im

def Equal_Points(center,*arounds):#返回与中心点像素相同的四周的点的个数
    number=0
    for i in arounds:
        if center==i:number+=1
    return number

def Vertical(img):#传入二值化后的图片进行垂直投影,返回像素矩阵每一列中黑点的个数
    pixdata=img.load()
    w,h=img.size
    #print(w,h)
    black_point_num=[]
    for x in range(w):
        black=0
        for y in range(h):
            if pixdata[x,y]==0:
                black+=1
        black_point_num.append(black)
    return black_point_num
def Level(img):#传入二值化后的图片进行水平投影,返回像素矩阵每一列中黑点的个数
    pixdata=img.load()
    w,h=img.size;black_point_num=[]
    for y in range(h):
        black=0
        for x in range(w):
            if pixdata[x,y]==0:
                black+=1
        black_point_num.append(black)
    return black_point_num

def Count_0(black_point_num):#返回black_point_num中第一个不为0的数的索引
    for i in black_point_num:
        if i!=0:
            num=black_point_num.index(i)
            break
    return num

def Crop_Vertical(img,th=1,maxlenth=65):#智能切割方法，
    #th=3表示垂直投影中连续3个0像素则切割，maxlenth表示允许输出的最大宽度，超过这个宽度则直接平分
    w,h=img.size
    child_img_list=[]
    vertical_point_num=Vertical(img)#垂直投影
    x_coordinate=Get_coordinate(vertical_point_num,th,maxlenth)
    for x in x_coordinate:
        child_img=img.crop((x[0],0,x[1],h))#垂直分割
        level_point_num=Level(child_img)#水平投影
        #y_coordinate=Get_coordinate(level_point_num)
        child_img=child_img.crop((0,Count_0(level_point_num),child_img.size[0],child_img.size[1]-Count_0(level_point_num[::-1])))#水平分割
        child_img=Fill_whitePoint(child_img)
        child_img_list.append(child_img)
    return child_img_list

def GetFeature(imgORpicpath,model):#从切割后的图片中获取特征
    if model=='path':img=Image.open(imgORpicpath)
    elif model=='img':img=imgORpicpath
    w,h=img.size
    pixel_cnt_list=[]
    for y in range(h):
        pix_cnt_x=0
        for x in range(w):
            if img.getpixel((x,y))==0:pix_cnt_x+=1
        pixel_cnt_list.append(pix_cnt_x)
    for x in range(w):
        pix_cnt_y=0
        for y in range(h):
            if img.getpixel((x,y))==0:pix_cnt_y+=1
        pixel_cnt_list.append(pix_cnt_y)
    return pixel_cnt_list

def GetFeature2(imgORpicpath,model):#从切割后的图片中获取特征
    if model=='path':img=Image.open(imgORpicpath)
    elif model=='img':img=imgORpicpath
    w,h=img.size
    pixel_cnt_list=[]
    for x in range(w):
        for y in range(h):
            pixel=img.getpixel((x,y))
            if pixel in [0,1,255]:
                if pixel==0:
                    pixel_cnt_list.append(pixel)
                else:
                    pixel_cnt_list.append(1)
            else:
                pixel_cnt_list.append(sum(pixel))
    return pixel_cnt_list

def GetFeatureStr(FeatureList,*Feature):#将整型特征列表转换为字符型列表
    if Feature:FeatureStr=['%s '%str(Feature[0])]
    else:FeatureStr=[]
    for i in range(len(FeatureList)):
        FeatureStr.append('%d:%d '%(i+1,FeatureList[i]))
    if Feature:FeatureStr.append('\n')
    else:FeatureStr=''.join(FeatureStr)
    return FeatureStr

def Get_coordinate(num,th=1,maxlenth=65):#传入垂直投影，返回x坐标
    front,behind=[],[]
    if num[0]!=0:front.append(0)
    if th==1:
        for i in range(len(num)):
            if i<len(num)-1:
                if (num[i],num[i+1]).count(0)==1:
                    if num[i]==0:
                        #print(i+1,num[i+1])
                        front.append(i+1)
                    else:
                        #print(i,num[i])
                        behind.append(i)
            if num[-1]!=0:behind.append(len(num)-1)
    elif th==3:
        if num[0]==0 and num[1]!=0:front.append(1)
        elif num[0]==0 and num[1]==0 and num[2]!=0:front.append(2)
        for i in range(len(num)):
            if i<len(num)-3:
                if (num[i],num[i+1],num[i+2],num[i+3]).count(0)==3:
                    if num[i]==0:
                        #print(i+1,num[i+1])
                        front.append(i+3)
                    else:
                        #print(i,num[i])
                        behind.append(i)
        if num[-1]!=0:behind.append(len(num)-1)
        elif num[-1]==0 and num[-2]!=0:behind.append(len(num)-2)
        elif num[-1]==0 and num[-2]==0 and num[-3]!=0:behind.append(len(num)-3)
    _coordinate=list(zip(front,behind))
    front,behind=[],[]
    for i in _coordinate:
        if 0<=minus(i)<=maxlenth:
            front.append(i[0])
            behind.append(i[1])
        elif maxlenth<minus(i)<=maxlenth*2:
            front.append(i[0])
            behind.append(math.floor(avg(i)))
            front.append(math.ceil(avg(i)))
            behind.append(i[1])
        elif maxlenth*2<minus(i)<=maxlenth*3:
            front.append(i[0])
            behind.append(math.floor(avg(i,3)[0]))
            front.append(math.ceil(avg(i,3)[0]))
            behind.append(math.floor(avg(i,3)[1]))
            front.append(math.ceil(avg(i,3)[1]))
            behind.append(i[1])
        else:
            front.append(i[0])
            behind.append(math.floor(avg(i,4)[0]))
            front.append(math.ceil(avg(i,4)[0]))
            behind.append(math.floor(avg(i,4)[1]))
            front.append(math.ceil(avg(i,4)[1]))
            behind.append(math.floor(avg(i,4)[2]))
            front.append(math.ceil(avg(i,4)[2]))
            behind.append(i[1])
    return list(zip(front,behind))

def Fill_whitePoint(im,th=5):#填充字符中的白点
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            if im.getpixel((x,y))==1:#如果是白色
                if around_black(im,x,y)>=th:#如果周围黑点个数大于等于5
                    im.putpixel((x,y),0)#将改点置为黑色
    return im
def around_black(img,x,y):#传入二值化后的图片，返回某个点周围黑点个数
    w,h=img.size
    n=0
    for coordinate in [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]:
        if (0<=coordinate[0]<w) and (0<=coordinate[1]<h):#判断是否为有效点
                if img.getpixel(coordinate)==0:n+=1
    return n

def Newimage(size):
    newimage=Image.new('L',size,255)
    newimage=newimage.point([0]*150+[1]*(256-150),'1')
    return newimage  

def Corrode(img,th=3):#腐蚀算法,传入二值化的图片
    w,h=img.size
    iCorrode=Newimage(img.size)
    for x in range(w):
        for y in range(h):
            num,eff=0,0#num:统计黑点个数，eff:统计有效点个数
            for coordinate in [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]:#8领域
                if (0<=coordinate[0]<w) and (0<=coordinate[1]<h):
                    eff+=1
                    num+=img.getpixel(coordinate)#统计领域中黑点个数,0代表黑色
            if num/eff<th/8:
                iCorrode.putpixel((x,y),img.getpixel((x,y)))
    iCorrode=de_noise(iCorrode)
    return iCorrode

def Expand(img,th=6):#膨胀算法,传入二值化的图片
    w,h=img.size
    iExpand=Newimage(img.size)
    for x in range(w):
        for y in range(h):
            num,eff=0,0#num:统计黑点个数，eff:统计有效点个数
            for coordinate in [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]:#8领域
                if (0<=coordinate[0]<w) and (0<=coordinate[1]<h):
                    eff+=1
                    num+=img.getpixel(coordinate)#统计领域中黑点个数,0代表黑色
            if num/eff<th/8:
                iExpand.putpixel((x,y),0)
    return iExpand

def upright(img):#返回旋转后宽度最短的图片
    criterion=100
    for i in range(-90,95,5):
        length=len(strip_list(Vertical(MyRotate(img,i))))
        if criterion>length:
            criterion=length
            angle=i
    upright_img=MyRotate(img,angle)
    vertical_point_num=Vertical(upright_img)#垂直投影
    x_coordinate=Get_coordinate(vertical_point_num,th=3)
    upright_img=upright_img.crop((x_coordinate[0][0],0,x_coordinate[-1][-1],upright_img.size[1]))
    return upright_img

def MyRotate(img,angle,expand=1):#旋转图片
    pilim=img.convert("RGBA").rotate(angle,3,expand=expand)
    newimg=Image.new('RGBA', pilim.size, (255,)*4)
    rotimg=Image.composite(pilim,newimg,pilim).convert('L').point([0]*150+[1]*(256-150),'1')
    return rotimg

def lstrip_list(lists,name=0):#列表的lstrip
    while 1:
        if lists[0]==name:
            lists.pop(0)
        else:break
    return lists

def rstrip_list(lists,name=0):
    while 1:
        if lists[-1]==name:
            lists.pop(-1)
        else:break
    return lists

def strip_list(lists,name=0):
    return rstrip_list(lstrip_list(lists,name),name)    

def ordd(strs):
    asc=''
    for i in strs:
        asc+=str(ord(i))
    return int(asc)    

def minus(num):#求两个元素的列表或tuple的差
    return abs(num[0]-num[1])

def avg(num,type=1):#求平均数,传入tuple或list
    if type==1:#1表示普通平均数
        return sum(num)/len(num)
    elif type==3:#3表示求两个数之间三分之一、三分之二的数
        minu=minus(num)
        one_third=num[0]+minu*(1/3)
        two_third=num[0]+minu*(2/3)
        return one_third,two_third
    elif type==4:#4表示求两个数之间四分之一、四分之二、四分之三的数
        minu=minus(num)
        one_quarter=num[0]+minu*(1/4)
        two_quarter=num[0]+minu*(2/4)
        three_quarter=num[0]+minu*(3/4)
        return one_quarter,two_quarter,three_quarter

def Rename(dpath):#识别目录下的验证码图片，并将识别结果作为图片名字
    pics=list(os.walk(dpath))[0][2]
    pics.sort(key=lambda x:ordd(x.split('.')[0]))
    path=list(os.walk(dpath))[0][0]
    for pic in pics:
        code=re.sub(r'\\|/|:|\*|\?|"|<|>|\|','_',GetCode_MC(os.path.join(path,pic)))
        newname='{}_{}.png'.format(code,str(uuid.uuid1()).replace('-','')[0:3])
        try:
            os.rename(os.path.join(path,pic),os.path.join(path,newname))
        except FileExistsError:
            newname='{}_{}.png'.format(code,str(uuid.uuid1()).replace('-',''))
            os.rename(os.path.join(path,pic),os.path.join(path,newname))
        sys.stdout.write('%s  \r'%pic)
        sys.stdout.flush()   
#####公共方法########end############公共方法########end############公共方法########end############公共方法########end############公共方法########end#######

if __name__=='__main__':
    #print(GetCode_login(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_login1\3a76.jpeg'))#3a76
    #print(GetCode_register(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_register1\2.jpg'))#kd2g
    #print(GetCode_bigboss(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_bigboss1\1.png'))#3489
    #print(GetCode_OM(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_OM1\1.jpeg'))#d27g
    # print(GetCode_MM(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_MM1\48.jpeg'))#pusr
    #print(GetCode_MC(r'E:\Users\ZP\Desktop\getVerifyCode.jpg'))
    # print(GetCode_SKB(r'E:\Users\ZP\Desktop\5-2\py\Training_SKB_phone\1_1e33.png'))
    print(GetCode_SKB_abc(r'E:\Users\ZP\Desktop\Tranning_SKB_abc\1_046662.png'))
    # print(GetCode_SKB_123(r'E:\Users\ZP\Desktop\5-2\py\Training_SKB_123\1_6188.png'))
    # print(GetCode_H5(r'E:\Users\ZP\Downloads\canvas.png'))
    #Rename(r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_MM2')