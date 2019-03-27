from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen("https://morvanzhou.github.io/static/scraping/list.html").read().decode("utf-8")
soup = BeautifulSoup(html, features='lxml')

#use class 
month = soup.find_all("li", {"class":"month"})  #二月的class属性是feb month也会匹配到
for m in month:
    print(m.get_text())


print("\n")
jan = soup.find("ul",{"class":"jan"})
days = jan.find_all("li")
for d in days:
    print(d.get_text())

