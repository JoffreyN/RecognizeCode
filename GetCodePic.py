import requests,os,sys,base64,json,re
requests.packages.urllib3.disable_warnings()#关闭ssl警告

def GetCodePic(url='http://116.228.151.160:43513/verify/getVerifyCode'):
    n=1
    path=os.path.join(os.getcwd(),'codes/codes_MC1')
    if not os.path.exists(path):os.mkdir(path)
    types=['jpg','jpeg','gif','png']
    while n<=1000:
        head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0'}
        r=requests.get(url,headers=head,verify=False)
        html=r.content
        ext=r.headers['Content-Type']
        #print(ext)
        for t in types:
            if t in ext.lower():
                suffix=t
                break
        picpath=os.path.join(path,f'{n}.{suffix}')
        with open(picpath, 'wb') as f:f.write(html)
        sys.stdout.write('%d  \r'%n)
        sys.stdout.flush()
        n+=1

# GetCodePic()

def Get_ali_codepic():
    n=1
    path=os.path.join(os.getcwd(),'codes/codes_ali')
    if not os.path.exists(path):os.mkdir(path)
    url='https://diablo.alibaba.com/captcha/click/get.jsonp?sessionid=01cwNTbEP8sZzvcWelxU2_IjP3FcKyXUtcv4vE8e3ev9ZtJvXNjbkokNyS9_hgwGoFsNIrzko6LorD66bCKKpIoyrtXYSbexK0ttkvTisFXG0&identity=QNYX&style=ncc&lang=cn&v=918&callback=jsonp_06974505904357461&t=0.2332687831095004'
    head={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0'}
    while n<=1000:
        r=requests.get(url,headers=head,verify=False)
        if r.text.startswith('jsonp'):
            picpath=os.path.join(path,f'{n}.jpg')
            jsdata=json.loads(re.sub(r'\(|\)','',re.findall(r'\(.+\)',r.text)[0]))
            picdata=base64.b64decode(jsdata['result']['data'][0].split(',')[1])
            with open(picpath, 'wb') as f:f.write(picdata)
            sys.stdout.write(f'{n}  \r')
            sys.stdout.flush()
            n+=1
        else:
            print(r.text)
            break
#Get_ali_codepic()