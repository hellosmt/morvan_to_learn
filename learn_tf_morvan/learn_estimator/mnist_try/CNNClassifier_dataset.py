import tensorflow as tf
import pandas as pd
from tensorflow import data


TRAIN_DATA_PATH = "/home/sunmengtuo/Workspace/morvan_to_learn/learn_tf_morvan/learn_estimator/mnist_try/data_csv/mnist_train.csv"
TEST_DATA_PATH = "~/Workspace/morvan_to_learn/learn_tf_morvan/learn_estimator/mnist_try/data_csv/mnist_test.csv"

IMG_SHAPE = [28, 28]
IMG_WIDTH = 28
IMG_HEIGHT = 28
NUM_CHANNEL = 1

LEARNING_RATE = 0.001

data_train = pd.read_csv(TRAIN_DATA_PATH, header=None) # DataFrame类型
data_test = pd.read_csv(TEST_DATA_PATH, header=None)
#print(type(data_test)) # DataFrame类型
#print(data_train)

train_values = data_train.values # values：numpy representation of NDFrame
#print(type(train_values)) # numpy ndarray类型
#print(train_values)
train_data = train_values[:,1:]/255.0 # 归一化处理
train_label = train_values[:,0:1].squeeze() # 所以这里可以进行切片操作
#print(train_label)

test_values = data_test.values
test_data = test_values[:,1:]/255.0
test_label = test_values[:,0:1].squeeze()


def my_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params["feature_columns"]) # 将特征字典和 feature_columns 转换为模型的输入层
    # 卷积需要4 rank，所以reshape
    net = tf.reshape(net, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL])
    net = tf.layers.conv2d(net, filters=16, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.conv2d(net, filters=36, kernel_size=5, padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(net, 2, 2)
    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, 128, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=10)
    pred = tf.nn.softmax(logits=logits)
    pred_cls = tf.argmax(pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        metrics = \
        {
            "accuracy": tf.metrics.accuracy(labels, pred_cls)
        }
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
    return spec



def _parse_line(line): # 将csv的一行数据转换成为tensor
    #COLUMNS = ['image']
    record_defaults = [[1.0] for col in range(785)] # Default values, in case of empty columns，防止有些值是空的
    #print(record_defaults)
    items = tf.decode_csv(line, record_defaults=record_defaults) #decode_csv 操作会解析这一行内容并将其转为张量列表
    #print(items)
    feature = items[1:785]
    label = items[0]
    feature = tf.cast(feature, tf.float64)
    feature = tf.reshape(feature, [28,28])
    #print(feature) # Tensor("Reshape:0", shape=(28, 28), dtype=float64)
    label = tf.cast(label, tf.int64)
    # feature = zip(COLUMNS, feature) # 使用内置函数构建字典 不能用dict？？说是要在eger模式下，不明白
    return feature, label


# 定义数据输入函数
def train_input_fn(csv_path):
    dataset = tf.data.TextLineDataset(csv_path).skip(1) # 调用 skip 方法来跳过文件的第一行，此行包含标题，而非样本，每一行就是dataset的一个元素
    #print(dataset) # <SkipDataset shapes: (), types: tf.string>
    dataset = dataset.map(_parse_line) # 对每一行调用_parse_line函数
    print(dataset) # <MapDataset shapes: ((28, 28, 1), ()), types: (tf.float64, tf.int64)>
    dataset = dataset.shuffle(buffer_size=10000)
    print(dataset)
    dataset = dataset.batch(1)  # 这里一定要用到batch，不然my_model_fn里面的reshape会报错，不知道为什么？
    print(dataset)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    # print(features)
    # print(labels)
    features = {"images":features}
    return features, labels

# 创建特征列，将其视为原始数据和Estimator之间的媒介，将各种原始数据转换为Estimator可以用的格式
feature_colume = tf.feature_column.numeric_column("images", shape=IMG_SHAPE) # 这个特征键值是image，shape为28x28，这里的特证名image要和原始数据的特征名字一致，否则模型无法找到特征数据
feature_columes = [feature_colume] # 这里只定义了一个特征，也就是这个图片，将所有特征放在一个list里面，这个特征列就是要传给Estimator进行构造的参数之一

#features, target = train_input_fn(TRAIN_DATA_PATH)
# print(features)
# print("Features in CSV: {}".format(list(features.keys())))
# print("Target in CSV: {}".format(target))


# 构造estimator
my_model = tf.estimator.Estimator(model_fn=my_model_fn, model_dir="./models", params={"feature_columns":feature_columes})
my_model.train(input_fn=lambda:train_input_fn(TRAIN_DATA_PATH), steps=2000)