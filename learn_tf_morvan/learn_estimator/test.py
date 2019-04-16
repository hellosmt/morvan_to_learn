'''使用预创建的Estimator，通常包含四个步骤：编写一个或多个数据集导入函数→定义特征列→实例化相关的预创建的Estimator→调用训练、评估或推# 理函数
'''
import tensorflow as tf 
# 数据集结构
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)
print(dataset1.output_shapes)

# 创建迭代器去访问数据集中的元素
# 单次迭代器：仅支持对数据集进行一次迭代，不需要显式初始化
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        assert i == value

# 可初始化迭代器：允许在初始化的时候使用一个或者多个tf.placeholder()去参数化数据集的定义
max_val = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_val)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_val:10})
    for i in range(10):
        value = sess.run(next_element)
        print(value)
