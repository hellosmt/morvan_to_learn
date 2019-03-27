from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen("https://morvanzhou.github.io/static/scraping/basic-structure.html").read().decode("utf-8")
soup = BeautifulSoup(html, features="lxml")
all_href = soup.find_all("a")   #获取到所有的标签<a></a> 每个a都是一个字典 href是它的一个属性，也就是这个字典里面的一个键
link = [i["href"]for i in all_href]  #然后我们遍历 找出所有的链接
print(link)

