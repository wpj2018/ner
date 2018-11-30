#coding:utf-8
import tensorflow as tf

INPUT_NODE = 100
OUTPUT_NODE = 7
MAX_SEQ_LEN = 100
HIDDEN_SIZE = 50
BATCH_SIZE = 128

def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, is_train=True, regularizer=None): #[batch, seqlen, emb_size]
    seq_lst = tf.transpose(x, [1, 0, 2])
    seq_len = 100
    if not is_train: #如果是训练集句长为100，为了防止内存不足，测试时句长为测试集中最长句长600
        seq_len = 600

    seq_lst = tf.unstack(seq_lst, num=seq_len) #将[batch, seqlen, emb_size]的输入，转化为seqlen个[batch, emb_size] 的list

    h = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32) #[batch, hidden_size]

    weight_xh = get_weight(shape=[INPUT_NODE, HIDDEN_SIZE]) #输入->隐藏层的权重矩阵
    bias_xh = get_bias(shape=[HIDDEN_SIZE]) #输入->隐藏层的偏置

    weight_hh = get_weight(shape=[HIDDEN_SIZE, HIDDEN_SIZE]) #隐藏层->隐藏层的权重矩阵，rnn特有
    bias_hh = get_bias(shape=[HIDDEN_SIZE]) #隐藏层->隐藏层的偏置

    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer) #隐藏层->输出层的权重矩阵
    bias_ho = get_bias(shape=[OUTPUT_NODE]) ##隐藏层->输出层的偏置

    seq_y = [] #seqlen 个[batch_size, output_node]
    with tf.variable_scope("SimpleRNN"):
        for i, inp in enumerate(seq_lst):
            if i > 0:
                tf.get_variable_scope().reuse_variables() #每一步，rnn的权值共享

            h = tf.matmul(inp, weight_xh) + bias_xh + tf.matmul(h, weight_hh) + bias_hh # h_t=f(x_t * W_xt + h_t-1 * W_hh)
            h = tf.sigmoid(h)

            y = tf.matmul(h, weight_ho) + bias_ho #[batch, OUTPUT_NODE]
            seq_y.append(y)
    return tf.transpose(seq_y, [1, 0, 2]) #转化为[batch_size, seqlen, output_node]



