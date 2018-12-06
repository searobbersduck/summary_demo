# !/usr/bin/env python3

import tensorflow as tf
import numpy as np

import sys
sys.path.append('../')
sys.path.append('./')
sys.path.append('../bert')
sys.path.append('../recommendrecord')


from bert.transformer import *
from params import *

def get_mask(inputs):
    # inputs:(N, T, C)
    mask = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
    mask = tf.ones_like(mask)
    # mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, inputs.get_shape().as_list()[-1]])
    return mask

class SummaryModel(object):
    def __init__(self, wordsEmbeddings, opts, is_training, global_step=None):
        resume_max_line = opts.resume_max_line
        resume_max_tokens_per_line = opts.resume_max_tokens_per_line
        summary_max_tokens_per_sent = opts.record_max_length
        positional_enc_dim = opts.positional_enc_dim
        dropout_rate = opts.dropout_rate
        num_heads = opts.num_heads
        num_layers = opts.num_layers
        hidden_size = opts.hidden_size
        resume_max_pool_size = opts.resume_max_pool_size
        vocab_size = opts.vocab_size
        num_sampled = opts.num_sampled
        learning_rate = opts.lr
        self.learning_rate = tf.placeholder(dtype=tf.float32,
                                       shape=None,
                                       name='lr')
        learning_rate = self.learning_rate
        batch_size = 4 # None, todo:
        self.resume_tokens = tf.placeholder(name='resumes',
                                            shape=[batch_size, resume_max_line, resume_max_tokens_per_line],
                                            dtype=tf.int32)
        self.summary_tokens = tf.placeholder(name='summaries',
                                             shape=[batch_size, summary_max_tokens_per_sent],
                                             dtype=tf.int32)
        self.words_embeddings = tf.convert_to_tensor(wordsEmbeddings)
        self.resume_tokens_emb = tf.nn.embedding_lookup(
            self.words_embeddings, self.resume_tokens
        )
        self.summary_tokens_emb = tf.nn.embedding_lookup(
            self.words_embeddings, self.summary_tokens
        )
        with tf.variable_scope('encoder'):
            self.raw_enc_shape = self.resume_tokens_emb.get_shape().as_list()
            self.enc = tf.reshape(
                self.resume_tokens_emb,
                shape=[-1, self.raw_enc_shape[2], self.raw_enc_shape[3]]
            )
            self.positional_enc = positional_encoding(
                self.enc,
                positional_enc_dim,
                scope='resume_positional_enc'
            )
            enc_mask = get_mask(self.enc)
            enc_mask = tf.tile(tf.expand_dims(enc_mask, axis=2), [1,1,self.positional_enc.get_shape().as_list()[-1]])
            self.enc = tf.concat([self.enc, self.positional_enc*enc_mask], axis=-1)
            self.enc = tf.layers.dropout(self.enc,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(is_training))
            for i in range(num_layers):
                with tf.variable_scope('num_block_{}'.format(i)):
                    self.enc = multihead_attention(
                        self.enc,
                        self.enc,
                        num_units=hidden_size,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        is_training=is_training,
                        causality=False
                    )
                    self.enc = feedforward(
                        self.enc, num_units=[hidden_size*4, hidden_size]
                    )
            self.enc = tf.reshape(
                self.enc,
                shape=[self.raw_enc_shape[0], -1, hidden_size]
            )
            self.enc = tf.transpose(self.enc, [0, 2, 1])
            self.enc = tf.layers.conv1d(
                inputs=self.enc,
                filters=resume_max_pool_size,
                kernel_size=1,
                activation=tf.nn.relu,
                use_bias=True
            ) #(N, hidden_size, resume_max_pool_size)
            self.enc = tf.transpose(self.enc, [0, 2, 1]) #(N, resume_max_pool_size, hidden_size)
        with tf.variable_scope('decorder'):
            self.raw_dec_shape = self.summary_tokens_emb.get_shape().as_list()
            self.dec = self.summary_tokens_emb
            self.positional_dec = positional_encoding(
                self.dec,
                positional_enc_dim,
                scope='summary_positional_enc'
            )
            self.positional_dec_mask = get_mask(self.dec)
            dec_mask = get_mask(self.dec)
            dec_mask = tf.tile(tf.expand_dims(dec_mask, axis=2), [1,1,self.positional_dec.get_shape().as_list()[-1]])
            self.dec = tf.concat([self.dec, self.positional_dec*dec_mask], axis=-1)
            self.dec = tf.layers.dropout(self.dec,
                                         rate=dropout_rate,
                                         training=tf.convert_to_tensor(is_training))
            for i in range(num_layers):
                with tf.variable_scope('num_block_{}'.format(i)):
                    self.dec = multihead_attention(
                        self.dec,
                        self.dec,
                        num_units=hidden_size,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        is_training=is_training,
                        causality=True, # todo:
                        scope='self_attention',
                        reuse=tf.AUTO_REUSE
                    )
                    self.dec = multihead_attention(
                        self.dec,
                        self.enc,
                        num_units=hidden_size,
                        num_heads=num_heads,
                        dropout_rate=dropout_rate,
                        is_training=is_training,
                        causality=False,
                        scope='vanilla_attention'
                    )
                    self.dec = feedforward(self.dec, num_units=[4*hidden_size, hidden_size])
                    # (N, summary_max_tokens_per_sent, hidden_size)

        # Final linear projection
        self.languge_model_weights = tf.get_variable(
            "languge_model_weights",
            shape=[vocab_size, hidden_size],
            regularizer=tf.contrib.layers.l2_regularizer(0.0001),
            initializer=tf.contrib.layers.xavier_initializer())
        self.languge_model_bias = tf.get_variable(
            "languge_model_bias", shape=[vocab_size])

        self.dec = tf.reshape(self.dec, shape=[-1, hidden_size])
        self.logits = tf.nn.xw_plus_b(self.dec, tf.transpose(self.languge_model_weights), self.languge_model_bias)
        # self.logits = tf.layers.dense(self.dec, vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=1))
        self.istarget = tf.to_int32(tf.not_equal(self.summary_tokens, 3))
        self.tmp = self.summary_tokens * self.istarget
        self.istarget = tf.to_float(tf.not_equal(self.tmp, 0))
        # self.istarget = tf.to_float(tf.not_equal(self.summary_tokens, 0))
        self.istarget = tf.reshape(self.istarget, shape=[-1])
        print(self.preds)
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, tf.reshape(self.summary_tokens, [-1])))*self.istarget)/tf.reduce_sum(self.istarget)
        tf.summary.scalar('acc', self.acc)
        if is_training:
            # loss
            self.loss = tf.nn.sampled_softmax_loss(
                self.languge_model_weights,
                self.languge_model_bias,
                tf.reshape(self.summary_tokens, [-1, 1]),
                tf.reshape(self.dec, [-1, hidden_size]),
                num_sampled,
                vocab_size
            )
            # self.loss = tf.reshape(self.loss, [-1, self.summary_tokens.get_shape().as_list()[-1]])
            self.loss = tf.reduce_sum(self.loss*self.istarget)/tf.reduce_sum(self.istarget)
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss += reg_losses

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients = [
                None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                for gradient in gradients
            ]
            self.train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            # tf.summary.scalar('loss', self.loss)

def test_SummaryModel():
    vocab_size = 8000
    hidden_size = 256
    wordsEmbeddings = tf.get_variable('wordsEmbeddings', shape=[vocab_size, hidden_size], dtype=tf.float32)
    opts = parse_args()
    model = SummaryModel(wordsEmbeddings, opts, True)
    print('hello world!')

def test_SummaryModel1():
    vocab_size = 8000
    hidden_size = 256
    opts = parse_args()
    # wordsEmbeddings = tf.get_variable('wordsEmbeddings', shape=[vocab_size, hidden_size], dtype=tf.float32)
    with tf.Graph().as_default():
        wordsEmbeddings = tf.get_variable('wordsEmbeddings', shape=[vocab_size, hidden_size], dtype=tf.float32)
        model = SummaryModel(wordsEmbeddings, opts, True)
    print('hello world!')

if __name__ == '__main__':
    test_SummaryModel1()







