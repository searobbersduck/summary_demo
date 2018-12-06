# !/usr/bin/env python3

import tensorflow as tf
import numpy as np

def layer_normalize(inputs):
    '''
    :param inputs: data need to layer normalize
    :return: the same size with inputs
    '''
    return tf.contrib.layers.layer_norm(inputs)


def positional_encoding(inputs, num_units, zero_pad=True, scale=True,
                        scope='positional_encoding', reuse=None):
    '''
    :param inputs:
    :param num_units:
    :param zero_pad:
    :param scale:
    :param scope:
    :param reuse:
    :return:
    '''
    N,T = inputs.get_shape().as_list()[:2]
    with tf.variable_scope(scope, reuse=reuse):
        position_index = tf.tile(tf.expand_dims(tf.range(T), axis=0),(N, 1)) #(N,T)
        position_enc = np.array([[pos/np.power(10000, 2.*i/num_units) for i in range(num_units)] for pos in range(T)]) #(T, num_units)
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) #position: 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) #position: 2i+1
        lookup_table = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=(1, num_units)), position_enc[1:]), axis=0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_index) #(N,T,num_units)
        if scale:
            outputs = outputs * num_units**0.5
        return outputs

def segment_encoding(a_len, ab_len, max_len, num_units,
                scope='segment_encoding', reuse=None):
    '''
    a_len = 5
    b_len = 7
    ab_len = 5+7 = 12
    mask_total = [2 2 2 2 2 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]

    :param a_len: the tokens length for sentence A
    :param ab_len: the tokens length for sentence B
    :param max_len: the max tokens length for one line
    :param num_units:
    :param scope:
    :param reuse:
    :return:
    '''
    mask_a = tf.sequence_mask(a_len, max_len, dtype=tf.int32) #(max_len)
    mask_ab = tf.sequence_mask(ab_len, max_len, dtype=tf.int32) #(max_len)
    mask_total = mask_a + mask_ab #(max_len)
    with tf.variable_scope(scope, reuse=reuse):
        lt_segment = tf.get_variable('lt_segment', shape=[3, num_units],
                                     initializer=tf.contrib.layers.xavier_initializer()) #(3, num_units)
        embeddings = tf.nn.embedding_lookup(lt_segment, mask_total) #(max_len, num_units)
    return embeddings #(max_len, num_units)

def multihead_attention(queries, keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope='multihead_attention',
                        reuse=None,
                        activation=None,
                        ):
    '''
    :param queries: a 3d tensor with shape of [N, Tq, Cq]
    :param keys: a 3d tensor with shape of [N, Tk, Ck]
    :param num_units:
    :param num_heads:
    :param dropout_rate:
    :param is_training:
    :param causality: Boolean. If true, If true, units that reference the future are masked.
    :param scope:
    :param reuse:
    :return: a 3d tensor with shape of [N, Tq, C], C=num_units
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]
        if activation is None:
            activation = tf.nn.relu
        # linear projection, C = num_units
        Q = tf.layers.dense(queries, num_units, activation=activation, name='Q') #(N, Tq, C)
        K = tf.layers.dense(keys, num_units, activation=activation, name='K') #(N, Tk, C)
        V = tf.layers.dense(keys, num_units, activation=activation, name='V') #(N, Tk, C)
        # split and concat, h = num_heads
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0) #(N*h, Tq, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0) #(N*h, Tk, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0) #(N*h, Tk, C/h)
        # keys
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) #(N*h, Tq, Tk)
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5) #(N*h, Tq, Tk)
        # keys mask
        keys_mask = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) #(N, Tk)
        keys_mask = tf.tile(keys_mask, [num_heads, 1]) #(N*h, Tk)
        keys_mask = tf.tile(tf.expand_dims(keys_mask, axis=1), [1, queries.get_shape().as_list()[1], 1])
        # keys mask outputs
        padding = tf.ones_like(outputs)*(-2**32+1) #(N*h, Tq, Tk)
        outputs = tf.where(tf.equal(keys_mask, 0), padding, outputs) #(N*h, Tq, Tk)
        # causality
        if causality:
            diag_vals = tf.ones_like(outputs[0]) #(Tq, Tk)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() #(Tq, Tk)
            masks = tf.tile(tf.expand_dims(tril, axis=0), [outputs.get_shape().as_list()[0], 1, 1]) #(N*h, Tq, Tk)
            padding = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), padding, outputs) #(N*h, Tq, Tk)
        # Activation
        outputs = tf.nn.softmax(outputs)
        # queries mask
        queries_mask = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) #(N, Tq)
        queries_mask = tf.tile(queries_mask, [num_heads, 1]) #(N*h, Tq)
        queries_mask = tf.tile(tf.expand_dims(queries_mask, axis=2), [1, 1, keys.get_shape().as_list()[1]]) #(N*h, Tq, Tk)
        outputs *= queries_mask #(N*h, Tq, Tk)
        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        #
        outputs = tf.matmul(outputs, V_) #(N*h, Tq, C/h)
        # restore
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=-1) #(N, Tq, C)
        # residual
        outputs += Q # todo: outputs += queries
        outputs = layer_normalize(outputs) #(N, Tq, C)
        return outputs


def feedforward(inputs, num_units=[2048, 512],
                scope='multihead_attention',
                reuse=None):
    '''
    :param inputs:
    :param num_units: [project_dim0, project_dim1]
    :param scope:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        outputs = tf.layers.conv1d(inputs=inputs, filters=num_units[0],
                                   kernel_size=1, activation=tf.nn.relu,
                                   use_bias=True)
        outputs = tf.layers.conv1d(inputs=outputs, filters=num_units[1],
                                   kernel_size=1, activation=tf.nn.relu,
                                   use_bias=True)
        outputs += inputs
        outputs = layer_normalize(outputs)
        return outputs



def label_smooth(inputs, epsilon=0.1):
    K = inputs.get_shape().as_list()[-1]
    return ((1-epsilon)*inputs)+(epsilon/K)


