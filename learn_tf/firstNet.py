
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  #可视化模块

def add_layer(inputs, layer_num, in_size, out_size,activation_function=None):
    layer_name = "layer%s"%layer_num
    with tf.name_scope("layer"):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name+'/Weights', Weights)  #我们想要看Weights变化的过程    这些都是histogram  但是loss不一样 loss是events  所以函数也不一样
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1, out_size])+0.1, name="b")
            tf.summary.histogram(layer_name+'/bias', bias) 
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs,Weights)+bias
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b) #relu默认有名字 所以不用定义
            tf.summary.histogram(layer_name+'/outputs', outputs) 
        return outputs

# 自己创的数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)  #设置噪声
y_data = np.square(x_data)-0.5 + noise

with tf.name_scope("input"):
    xs = tf.placeholder(tf.float32, [None, 1], name = "x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name = "y_input")

layer1 = add_layer(xs, 1, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 2, 10, 1,activation_function = None)

with tf.name_scope("losses"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]), name = "loss")
    tf.summary.scalar("loss", loss)  #loss的训练图和weights bias不一样 它是event

with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
### 给所有的训练图打包
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log/", sess.graph) #一定要在定义了sess之后再write 这里就是把所有的信息 框架收集起来写到一个文件里面去 然后我们才能在浏览器里看见它
sess.run(init)

#### plt可视化定义
# fig = plt.figure() #生成一个图片框
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data, y_data)  #这个是用点的形式plot上去
# plt.ion()  
# plt.show() #这个show是plot了一次之后程序就暂停了 要让它不暂停 不停地去plot就要加上plt.ion() 新版Python是这样 老板是plt.show(block=False)

for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))  #当运算要用到placeholder时 都要用到feed_dict
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs, i)


    #    # plt可视化
    #    try:#先抹除 再plot  这样衔接的好一些 try是因为第一次的时候还没有lines 所以会报错
    #        ax.lines.remove(lines[0])  #去除掉画的线 不然会有很多根线画在上面
    #    except Exception:
    #        pass
    #    prediction_value = sess.run(prediction, feed_dict={xs:x_data})
    #    lines = ax.plot(x_data, prediction_value, 'r-', lw=5) #red 线宽5
       
    #    plt.pause(0.1) #暂停0.1s再继续