###实现去噪自编码器（给数据加上噪声）

import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

##参数初始化方法Xavier initialization，定义标准的均匀分布的Xavier初始化器，
# 它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
##通过tf.random_uniform创建一个均匀分布
##fan_in,fan_out 分别是输入、输出的节点数量
def xavier_init(fan_in , fan_out , constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),
                             minval = low , maxval = high,
                             dtype = tf.float32
                             )

##定义一个去噪自编码的class，包括构建函数以及一些其他调用函数
##类中包含一个构建函数 _init_() ,_init_()中包含几个输入：n_input为输入变量数， n_hidden为隐藏层节点数，
##transfer_function为隐藏层激活函数，默认为softplus，优化器optimizer默认为Adam，高斯噪声参数scale默认为0.1
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self ,n_input ,n_hidden ,transfer_function = tf.nn.softplus ,
                 optimizer = tf.train.AdamOptimizer , scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

 ##接下来定义网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])    ##None表示未知,不给定具体的值，由输入的数目来决定
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x +scale * tf.random_normal((n_input,)),
            self.weights['w1'] ), self.weights['b1']))
        ##在输入的数据上加上高斯噪声，并使用self.transfer对结果进行激活函数处理
        self.reconstruction = tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        ##输出层进行数据复原，重建操作（即建立reconstruction层）

##定义自编码器的损失函数，直接使用平方误差作为cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))   ##
        self.optimizer = optimizer.minimize(self.cost)    ##定义训练操作为优化器self.optimizer对损失函数self.cost的最小化
        init = tf.global_variables_initializer()         ##创建session，并初始化自编码器的全部模型参数
        self.sess = tf.Session()
        self.sess.run(init)

##参数初始化函数_initialize_weights()
    def _initialize_weights(self):
        all_weights = dict()         ##创建一个字典
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        #w1使用xavier_init函数初始化，传入输入节点数与隐藏层节点数，即可得到一个比较适合softplus等激活函数的初始状态
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]),dtype=tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden , self.n_input]), dtype=tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]), dtype=tf.float32)
        return all_weights

##定义计算cost及执行一步训练的的函数partial_fit()，即用一个batch数据进行训练并返回当前的损失cost
    def partial_fit(self,X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x:X, self.scale: self.training_scale} )
        return cost
##只求cost的函数，在自编码器的训练操作结束后，在训练集上对模型性能进行评测时用到
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x:X , self.scale: self.training_scale})

##定义transform函数，用以返回自编码器隐藏层的输出结果
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x:X , self.scale: self.training_scale})

##定义generate函数，将隐藏层的输出作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

##定义reconstruct函数，整体运行一遍复原过程，包括提取高阶特征（transform）和通过高阶特征复原数据（generate）
#  输入是原数据，输出为复原后的数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x:X, self.scale:self.training_scale})
##获取隐藏层的权重w1
    def get_weights(self):
        return self.sess.run(self.weights['w1'])
##获取隐藏层的偏置b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

##对mnist数据集进行性能测试
mnist = input_data.read_data_sets("C:/MNIST_data/",one_hot=True)

##定义对数据进行标准化处理的函数  可以直接使用sklearn中的类StandardScaler
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test
##定义一个获取随机block数据的函数，属于不放回抽样
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)   ##起始位置为随机数
    return data[start_index : (start_index + batch_size)]    ##返回从start_index开始的，长度为batch_size的一串数据

X_train,X_test = standard_scale(mnist.train.images, mnist.test.images)   ##对训练集与测试集进行标准化变换

n_samples = int(mnist.train.num_examples)   ##设置总训练样本数
training_epoches = 20     ##最大训练轮数
batch_size = 128
display_step = 1   ##训练一轮就显示一次cost

##创建一个AGN自编码器实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

for epoch in range(training_epoches):
    avg_cost = 0.
    total_batch = int(n_samples/batch_size)    ##总共需要的batch数，即为样本总数除以每次的训练batch_size数
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)    ##在每个batch循环中，随机抽取一个block数据
        cost = autoencoder.partial_fit(batch_xs)     ##然后训练这个batch数据，并计算当前cost
        avg_cost += cost/n_samples * batch_size     ##计算出平均损失
    if epoch%display_step == 0:
        print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))

print("Total cost:"+str(autoencoder.calc_total_cost(X_test)))   ##对测试集进行测试