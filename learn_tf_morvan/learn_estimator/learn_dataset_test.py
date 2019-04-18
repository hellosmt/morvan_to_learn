'''使用预创建的Estimator，通常包含四个步骤：编写一个或多个数据集导入函数→定义特征列→实例化相关的预创建的Estimator→调用训练、评估或推# 理函数
'''
import tensorflow as tf 
# 数据集结构
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10])) # 切分它形状上的第一个维度，即dataset中每一个元素是矩阵的一行，共有四个元素
print(dataset1.output_types)
print(dataset1.output_shapes)

# 创建迭代器去访问数据集中的元素
# 单次迭代器：仅支持对数据集进行一次迭代，不需要显式初始化，不支持参数化，数据消费完就抛出OutOfRangeError
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(100):
        value = sess.run(next_element)
        assert i == value

# 可初始化迭代器：允许在初始化的时候使用一个或者多个tf.placeholder()去参数化数据集的定义，实现了单个iterator下单个dataset中填充数据的切换
max_val = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_val)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={max_val:10})
    for i in range(10):
        value = sess.run(next_element)
        print(value)

# 可重新初始化迭代器：实现单个iterator下多个dataset的切换
training_data = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64)) # 对训练数据有一个随机干扰
validation_data = tf.data.Dataset.range(50)
iterator = tf.data.Iterator.from_structure(training_data.output_types, training_data.output_shapes) # 通过结构构造，训练和测试数据结构相同，指定哪个都可以
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_data) 
validation_init_op = iterator.make_initializer(validation_data)

with tf.Session() as sess:
    for _ in range(20):
        sess.run(training_init_op)
        for _ in range(100):
            sess.run(next_element)

        sess.run(validation_init_op)
        for _ in range(50):
            sess.run(next_element)    

# 可馈送迭代器：多个iterator下多个dataset的切换 feed的是不同迭代器的的handle
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# The `Iterator.string_handle()` method returns a tensor that can be evaluated
# and used to feed the `handle` placeholder.
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

with tf.Session() as sess:
    # Loop forever, alternating between training and validation.
    while True:
        # Run 200 steps using the training dataset. Note that the training dataset is
        # infinite, and we resume from where we left off in the previous `while` loop
        # iteration.
        for _ in range(200):
            sess.run(next_element, feed_dict={handle: training_handle})

        # Run one pass over the validation dataset.
        sess.run(validation_iterator.initializer)
        for _ in range(50):
            sess.run(next_element, feed_dict={handle: validation_handle})

# 要在 input_fn 中使用 Dataset（input_fn 属于 tf.estimator.Estimator），只需返回 Dataset 即可，框架将负责为您创建和初始化迭代器。