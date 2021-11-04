import requests
url=' http://127.0.0.1:5000/'
files={'file': open('4 (1).jpg','rb')}
r=requests.post(url,files=files)

print(r.text)