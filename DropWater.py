import os
from PIL import Image
from GetCode import *

def Appetizer(picpath):#前期处理，二值化、降噪等
	im=Image.open(picpath).convert('L')#打开图片，并转化为灰度图
	im=im.point([0]*150+[1]*(256-150),'1')#二值化
	#im.show()
	im=de_noise(im)#降噪
	#im.show()
	return im
#Appetizer('./codes/76.jpeg')

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
#print(Vertical(Appetizer('./codes/21.jpeg')))

def DropWater(img):#滴水算法切割图片
	pass

def Count_0(black_point_num):#计算black_point_num中开头的0的个数
	for i in black_point_num:
		if i!=0:
			num=black_point_num.index(i)
			break
	return num




'''
def getall0num(dpath):#将目录下的验证码图片切割并保存到Training目录,即创建训练图片，之后需要手动分类
	pics=list(os.walk(dpath))[0][2]
	headnum,endnum=[],[]
	for pic in pics:
		picpath=os.path.join(dpath,pic)
		black_point_num=Vertical(Appetizer(picpath))
		headnum.append(Count_0(black_point_num))
		endnum.append(Count_0(black_point_num[::-1]))
	print('headnum',headnum)
	print('endnum',endnum)
getall0num('./codes')
'''

if __name__ == '__main__':
	print(Vertical(Appetizer('./codes/21.jpeg')))