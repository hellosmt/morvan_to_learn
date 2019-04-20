import tensorflow as tf
import pandas as pd
from tensorflow import data


TRAIN_DATA_PATH = "~/Workspace/morvan_to_learn/learn_tf_morvan/learn_estimator/mnist_try/data_csv/mnist_train.csv"
TEST_DATA_PATH = "~/Workspace/morvan_to_learn/learn_tf_morvan/learn_estimator/mnist_try/data_csv/mnist_test.csv"

IMG_SHAPE = [28, 28]

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


mnist_ds = tf.data.Dataset.from_tensor_slices(train_data)
print(mnist_ds)

def _parse_line(line): # 将csv的一行数据转换成为tensor
    record_defaults = [[1.0] for col in range(785)] # Default values, in case of empty columns.
    print(record_defaults)
    items = tf.decode_csv(line, record_defaults=record_defaults) #decode_csv 操作会解析这一行内容并将其转为张量列表
    feature = items[1:785]
    label = items[0]
    feature = tf.cast(feature, tf.float64)
    feature = tf.reshape(feature, IMG_SHAPE)
    label = tf.cast(label, tf.int64)
    return feature, label


# 定义数据输入函数
def train_input_fn(csv_path):
    dataset = tf.data.TextLineDataset(csv_path).skip(1) # 调用 skip 方法来跳过文件的第一行，此行包含标题，而非样本
    dataset = dataset.map(_parse_line) # 对每一行调用_parse_line函数
    
    return dataset

# 创建特征列，将其视为原始数据和Estimator之间的媒介，将各种原始数据转换为Estimator可以用的格式
feature_colume = tf.feature_column.numeric_column("image", shape=IMG_SHAPE) # 这个特征键值是image，shape为28x28，这里的特证名image要和原始数据的特征名字一致，否则模型无法找到特征数据
feature_columes = [feature_colume] # 这里只定义了一个特征，也就是这个图片，将所有特征放在一个list里面，这个特征列就是要传给Estimator进行构造的参数之一
train_input_fn(TRAIN_DATA_PATH)