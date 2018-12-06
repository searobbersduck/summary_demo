# !/usr/bin/env python3

import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')

from params import *
from bert.transformer import *
from layers0 import optimized_trilinear_for_attention, mask_logits

class RecommendRecordModel(object):
    def __init__(self,
                 wordsEmbeddings,
                 opts,
                 is_training
                 ):
        resume_max_line = opts.resume_max_line
        resume_max_tokens_per_line = opts.resume_max_tokens_per_line
        jd_max_line = opts.jd_max_line
        jd_max_tokens_per_line = opts.jd_max_tokens_per_line
        record_max_length = opts.record_max_length
        batch_size = 64 # None
        self.resume_tokens = tf.placeholder(dtype=tf.int32,
                                       shape=[batch_size, resume_max_line, resume_max_tokens_per_line],
                                       name='resumes')
        self.jd_tokens = tf.placeholder(dtype=tf.int32,
                                   shape=[batch_size, jd_max_line, jd_max_tokens_per_line],
                                   name='jds')
        self.record_target = tf.placeholder(dtype=tf.int32,
                                       shape=[batch_size, record_max_length])
        self.words_embedding = tf.convert_to_tensor(wordsEmbeddings, dtype=tf.float32)
        self.resume_tokens_embedding = tf.nn.embedding_lookup(
            self.words_embedding, self.resume_tokens
        ) #(N, Lr, Tr, Cr)
        self.jd_tokens_embedding = tf.nn.embedding_lookup(
            self.words_embedding, self.jd_tokens
        ) #(N, Lj, Tj, Cj)
        self.record_target_embedding = tf.nn.embedding_lookup(
            self.words_embedding, self.record_target
        ) #(N, Trt, Crt), rt: recommend record target
        # reshape
        with tf.variable_scope('encoder'):
            self.raw_resume_shape = self.resume_tokens_embedding.get_shape().as_list()
            self.raw_jd_shape = self.jd_tokens_embedding.get_shape().as_list()
            self.resume_tokens_embedding = tf.reshape(
                self.resume_tokens_embedding,
                shape=[-1, self.raw_resume_shape[1]*self.raw_resume_shape[2], self.raw_resume_shape[3]]
            )
            self.jd_tokens_embedding = tf.reshape(
                self.jd_tokens_embedding,
                shape=[-1, self.raw_jd_shape[1]*self.raw_jd_shape[2], self.raw_jd_shape[3]]
            )
            # positional encoding
            self.resume_positional_enc = positional_encoding(
                self.resume_tokens_embedding,
                opts.positional_enc_dim,
                scope='resume_positional_enc',
            )
            self.resume_enc = tf.concat(
                (self.resume_tokens_embedding, self.resume_positional_enc),
                axis=-1
            )
            self.jd_positional_enc = positional_encoding(
                self.jd_tokens_embedding,
                opts.positional_enc_dim,
                scope='jd_positional_enc'
            )
            self.jd_enc = tf.concat(
                (self.jd_tokens_embedding, self.jd_positional_enc),
                axis=-1
            )
            self.resume_enc = tf.layers.dropout(self.resume_enc,
                                                rate=opts.dropout_rate,
                                                training=tf.convert_to_tensor(is_training))
            with tf.variable_scope('resume_encoder'):
                # (N, Lr*Tr, Cr), Cr=positional_enc_dim + words_embedding_size
                self.resume_enc = tf.reshape(
                    self.resume_enc,
                    shape=[self.raw_resume_shape[0]*self.raw_resume_shape[1],
                           self.raw_resume_shape[2],
                           (self.raw_resume_shape[3]+opts.positional_enc_dim)]
                )
                for i in range(opts.num_layers):
                    with tf.variable_scope('num_block_{}'.format(i)):
                        self.resume_enc = multihead_attention(
                            self.resume_enc,
                            self.resume_enc,
                            num_units = opts.hidden_size,
                            num_heads = opts.num_heads,
                            dropout_rate = opts.dropout_rate,
                            is_training = is_training,
                            causality=False
                        )
                        self.resume_enc = feedforward(
                            self.resume_enc,
                            num_units=[opts.hidden_size*4, opts.hidden_size]
                        )
                # (N*Lr, Tr, Cr), Cr=hidden_size
            with tf.variable_scope('jd_encoder'):
                # (N, Lj*Tj, Cj), Cj=positional_enc_dim + words_embedding_size
                self.jd_enc = tf.reshape(
                    self.jd_enc,
                    shape=[self.raw_jd_shape[0]*self.raw_jd_shape[1],
                           self.raw_jd_shape[2],
                           (self.raw_jd_shape[3]+opts.positional_enc_dim)]
                )
                for i in range(opts.num_layers):
                    with tf.variable_scope('num_block_{}'.format(i)):
                        self.jd_enc = multihead_attention(
                            queries=self.jd_enc,
                            keys = self.jd_enc,
                            num_units = opts.hidden_size,
                            num_heads = opts.num_heads,
                            dropout_rate = opts.dropout_rate,
                            is_training = is_training,
                            causality=False
                        )
                        self.jd_enc = feedforward(
                            self.jd_enc,
                            num_units=[opts.hidden_size*4, opts.hidden_size]
                        )
                # (N*Lj, Tj, Cj), Cj=hidden_size
            with tf.variable_scope('resume_jd_encoder'):
                self.resume_enc = tf.reshape(
                    self.resume_enc,
                    shape=[self.raw_resume_shape[0], self.raw_resume_shape[1] * self.raw_resume_shape[2],
                           self.raw_resume_shape[3]]
                )
                self.resume_enc = tf.transpose(self.resume_enc, [0, 2, 1])
                self.jd_enc = tf.reshape(
                    self.jd_enc,
                    shape=[self.raw_jd_shape[0], self.raw_jd_shape[1] * self.raw_jd_shape[2], self.raw_jd_shape[3]]
                )
                self.jd_enc = tf.transpose(self.jd_enc, [0, 2, 1])
                self.resume_enc = tf.layers.conv1d(
                    inputs=self.resume_enc,
                    filters=opts.resume_max_pool_size,
                    kernel_size=1, activation=tf.nn.relu,
                    use_bias=True
                ) #(N, resume_max_pool_size, hidden_size)
                self.jd_enc = tf.layers.conv1d(
                    inputs=self.jd_enc,
                    filters=opts.jd_max_pool_size,
                    kernel_size=1, activation=tf.nn.relu,
                    use_bias=True
                ) #(N, jd_max_pool_size, hidden_size)
                self.resume_enc = tf.transpose(self.resume_enc, (0,2,1))
                self.jd_enc = tf.transpose(self.jd_enc, (0,2,1))
                self.resume_enc_mask = tf.cast(self.resume_enc, tf.bool)
                self.resume_enc_mask = tf.cast(self.resume_enc_mask, tf.int32)
                self.resume_enc_mask_01 = tf.reduce_max(self.resume_enc_mask, axis=2)
                self.resume_enc_maxlen = tf.reduce_max(tf.reduce_sum(self.resume_enc_mask_01, axis=1))
                self.jd_enc_mask = tf.cast(self.jd_enc, tf.bool)
                self.jd_enc_mask = tf.cast(self.jd_enc_mask, tf.int32)
                self.jd_enc_mask_01 = tf.reduce_max(self.jd_enc_mask, axis=2)
                self.jd_enc_maxlen = tf.reduce_max(tf.reduce_sum(self.jd_enc_mask_01, axis=1))
                S = optimized_trilinear_for_attention([self.resume_enc, self.jd_enc],
                                                      opts.resume_max_pool_size,
                                                      opts.jd_max_pool_size,
                                                      input_keep_prob=1.0 - opts.dropout_rate)
                mask_jd = tf.expand_dims(self.jd_enc_mask_01, 1)
                S_ = tf.nn.softmax(mask_logits(S, mask=mask_jd))
                mask_resume = tf.expand_dims(self.resume_enc_mask_01, 2)
                S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask=mask_resume), dim=1), (0, 2, 1))
                self.c2q = tf.matmul(S_, self.jd_enc)
                self.q2c = tf.matmul(tf.matmul(S_, S_T), self.resume_enc)
                attention_outputs = [self.resume_enc,
                                     self.c2q,
                                     self.resume_enc * self.c2q,
                                     self.resume_enc * self.q2c] #
        with tf.variable_scope('decoder'):
            self.record_target_positional_enc = positional_encoding(
                self.record_target_embedding,
                opts.positional_enc_dim,
                scope='record_target_positional_enc',
            )
            self.record_target_dec = tf.concat(
                [self.record_target_embedding, self.record_target_positional_enc],
                axis=-1
            ) #(N, Trt, Crt), Crt = words_embedding_size + positional_enc_dim
            self.record_target_dec = tf.layers.dropout(
                self.record_target_dec, rate=opts.dropout_rate,
                training=tf.convert_to_tensor(is_training)
            )
            for i in range(opts.num_layers):
                self.record_target_dec = multihead_attention(
                    queries=self.record_target_dec,
                    keys=self.record_target_dec,
                    num_units=opts.hidden_size,
                    num_heads=opts.num_heads,
                    dropout_rate=opts.dropout_rate,
                    is_training=is_training,
                    causality=True,
                    scope='self_attention'
                )
                self.record_target_dec = multihead_attention(
                    queries=self.record_target_dec,
                    keys=self.record_target_dec,
                    num_units=opts.hidden_size,
                    num_heads=opts.num_heads,
                    dropout_rate=opts.dropout_rate,
                    is_training=is_training,
                    causality=True,
                    scope='self_attention'
                )
            print('hello world!')


def test_RecommendRecordModel():
    vocab_size=1000
    embeddings_dim=256
    wordsEmbeddings = tf.get_variable('wordsEmbeddings', shape=[vocab_size, embeddings_dim], dtype=tf.float32)
    is_training=True
    opts = parse_args()
    print('====> opts: ')
    print(opts)
    RecommendRecordModel(wordsEmbeddings, opts=opts, is_training=is_training)
    print('hello world!')

if __name__ == '__main__':
    test_RecommendRecordModel()







