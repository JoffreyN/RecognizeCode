import urllib.request,os

def GetCodePic(url):
    n=1
    path=os.path.join(os.getcwd(),'codes')
    if not os.path.exists(path):os.mkdir(path)
    types=['jpg','jpeg','gif','png']
    while n<=500:
        request=urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:53.0) Gecko/20100101 Firefox/53.0')
        response=urllib.request.urlopen(request)
        html=response.read()
        ext=response.headers['Content-Type']
        for t in types:
            if t in ext.lower():
                suffix=t
                break
        picpath=os.path.join(path,'%d.%s'%(n,suffix))
        with open(picpath, 'wb') as f:f.write(html)
        n+=1

GetCodePic('https://passport.55188.com/index/validate/reg_mobile')
