import requests
from bs4 import BeautifulSoup

res = requests.get('https://sites.google.com/view/gaea-version')
soup = BeautifulSoup(res.content, 'html.parser')
title = soup.find('h1', attrs = {'id': 'h.xh216tc7v0ru', 'dir':'ltr', 'class':'zfr3Q duRjpb'})

if "1.0.0" == title.get_text() :
    print("일치")
print(title.get_text())