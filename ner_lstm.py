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
    seq_len = 100
    if not is_train:
        seq_len = 600
    '''
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
    output, _ = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)

    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer)
    bias_ho = get_bias(shape=[OUTPUT_NODE])

    y = tf.matmul(tf.reshape(output, [-1, HIDDEN_SIZE]), weight_ho) + bias_ho
    return tf.reshape(y, [-1, seq_len, OUTPUT_NODE])
    '''

    seq_lst = tf.transpose(x, [1, 0, 2])
    seq_lst = tf.unstack(seq_lst, num=seq_len)

    W_i = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])

    W_f = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])

    W_o = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])

    W_c = get_weight([INPUT_NODE + HIDDEN_SIZE, HIDDEN_SIZE])

    b_i = get_bias(HIDDEN_SIZE)

    b_f = get_bias(HIDDEN_SIZE)

    b_o = get_bias(HIDDEN_SIZE)

    b_c = get_bias(HIDDEN_SIZE)

    def lstm(x, h, c):
        input = tf.concat([h, x], -1)
        i = tf.sigmoid(tf.matmul(input, W_i) + b_i)
        f = tf.sigmoid(tf.matmul(input, W_f) + b_f)
        o = tf.sigmoid(tf.matmul(input, W_o) + b_o)

        candidate = tf.tanh(tf.matmul(input, W_c)) + b_c
        c = f * c + i * candidate
        h = o * tf.tanh(c)
        return h, c

    weight_ho = get_weight(shape=[HIDDEN_SIZE, OUTPUT_NODE], regularizer=regularizer)
    bias_ho = get_bias(shape=[OUTPUT_NODE])

    seq_y = []
    h = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32)
    c = tf.zeros(shape=[tf.shape(x)[0], HIDDEN_SIZE], dtype=tf.float32)

    with tf.variable_scope("SimpleRNN"):
        for i, inp in enumerate(seq_lst):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            h, c = lstm(inp, h, c)

            y = tf.matmul(h, weight_ho) + bias_ho #[batch, OUTPUT_NODE]
            seq_y.append(y)
    return tf.transpose(seq_y, [1, 0, 2])




