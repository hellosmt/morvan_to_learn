'''
教程参照：https://www.jianshu.com/p/1da8db98a073 https://www.jianshu.com/p/0761906c4c04
'''
import pandas as pd
import numpy as np

#Series 是一种类，用于表示1维带标签的数组s,第一个参数为data，第二个参数为label
S_with_user_label = pd.Series([0,1,2,3,4,5], ['a','b','c','d','e','f'])
S_with_default_label = pd.Series([2,4,5,9,8,7])
print(S_with_user_label)
print(S_with_default_label)
print(S_with_user_label.values) # 有values和index
print(S_with_user_label.index)

# 也可以用字典来创建
d = {'b':1, 'c':2, 'a':3}
S_from_dict = pd.Series(d) # 没有指定index，则按照键值进行排序 
print(S_from_dict) # 这里和预期的不一样啊啊啊啊？难道不应该按照abc的顺序排列吗
S_from_dict_1 = pd.Series(d, ['c', 'a', 'b']) # 指定了index，则按照index顺序进行排序
print(S_from_dict_1)

# 用标量创建
S_from_scalar = pd.Series(5, ['a', 'b', 'c'])
print(S_from_scalar)

# 和numpy结合
random_data = np.random.rand(1,5) # 均匀分布在0,1之间
S = pd.Series(random_data[0])
print(S[:2]) 
# Series对象，可以作为大多数numpy函数的输入参数，因为这些函数的输入参数类型都是array_like object
print(np.exp(S))
print(np.sin(S))

# 使用键值引用
print(S_from_scalar['a'])
S_from_scalar['a'] = 10
print(S_from_scalar)
# 检查有没有d
print('d' in  S_from_scalar)

# Series对象间做加减乘除，只有相同label对应的值才会进行运算
random_data_new = np.linspace(1,3,3)
S1 = pd.Series(random_data_new, ['a', 'b', 'c'])
S2 = pd.Series(random_data_new, ['e', 'f', 'g'])
print(S1+S1)
print(S1+S2)
print(S2*3)