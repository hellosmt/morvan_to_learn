'''
教程参照：https://www.jianshu.com/p/8505c880a3fb
'''

import pandas as pd

# 把DataFrame想象成一个电子表格，它由行名（index）、列名(columns)和数据(values)组成,在Pandas中，DataFrame类的列（column）类型是Series，又可以把DataFrame看做Series的列表（list）
# 创建方式一:由Series作为键值的字典创建,ABC相当于列名，123相当于行名
d = {'A': pd.Series([1,4,7], ['1','2','3']),'B':pd.Series([2,5,8], ['1','2','3']), 'C':pd.Series([9,7,5], ['1','2','3'])}
df = pd.DataFrame(d)
print(df)
print(df.index)
print(df.columns)
print(df.values)

# 创建方式二：由list作为键值字典传入
d = {'A':[1,4,8], 'B':[2,5,9], 'C':[8,7,9]}
df = pd.DataFrame(d, index=['1', '2', '3'])
print(df)

# 创建方式三，由二维NumPy ndarray类型创建 没看太明白？