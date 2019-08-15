from selenium import webdriver
from tools import out
import os,uuid,base64

def getImgData(browser):
	browser.find_element_by_id('ImgCanvas').click()
	code_img=browser.find_element_by_id('base64_img').text.split(',')
	return code_img[0],code_img[2]

def savePic(code,imgdata,fpath=r'E:\Users\ZP\Desktop\5-2\py\RecognizeCode\codes\codes_H5',n=0):
	if not os.path.exists(fpath):os.mkdir(fpath)
	picName=f'{code}_{uuid.uuid1().hex[:8]}.png'
	picPath=os.path.join(fpath,picName)
	with open(picPath,'wb') as f:f.write(base64.b64decode(imgdata))
	out(f"{n} {picName}")

if __name__ == '__main__':
	browser=webdriver.Firefox()
	browser.set_window_size(400,700)
	browser.get('http://192.168.4.47/yam.html')
	for i in range(100):
		code,imgdata=getImgData(browser)
		savePic(code,imgdata,n=i)
	browser.quit()