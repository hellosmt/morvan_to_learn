import re

pattern1 = r"r[ua]n"

string = "Cat runs to Cat "
print(re.search(pattern1, string))
print(re.search(r"r[A-Z]n", string))

print(re.search(r"Mon(day)?", "Monday"))


str = """nihao 
smt"""
print(re.search(r"^s",str, flags=re.M))


print(re.search(r"abb+", "ab"))


string1 = "id:20180304, Date:2019-08-15 14:45"
match = re.search(r"(?P<id>\d+), Date(?P<date>.+)",string1)
print(match.group())
print(match.group("id"))
print(match.group(2))
