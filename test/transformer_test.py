# !/usr/bin/env python3

import os
import sys
import tensorflow as tf

sys.path.append('../')

from bert.transformer import *

import unittest
from parameterized import parameterized

class Test_Transformer(unittest.TestCase):
    @parameterized.expand([
        (2, 3, 4),
        (3, 4, 5),
        (64, 20, 512),
        (32, 30, 1024),
    ])
    def test_layer_normalize(self, dim_0, dim_1, dim_2):
        shape1 = [dim_0, dim_1, dim_2]
        input_c = tf.ones(shape1)
        inputs = tf.Variable(input_c)
        outputs = layer_normalize(inputs)
        self.assertEqual(outputs.get_shape(), shape1)

    def test_positional_encoding(self):
        N=64
        Tq=200
        Tk=200
        queries = tf.placeholder(shape=[N, Tq], dtype=tf.float32)
        keys = tf.placeholder(shape=[N, Tk], dtype=tf.float32)
        num_units = 512
        outputs = positional_encoding(queries, num_units)
        self.assertEqual(outputs.get_shape(), [N, Tq, num_units])

    def test_segment_encoding(self):
        alen = 5
        blen = 10
        maxlen = 30
        outputs = segment_encoding(alen, blen, maxlen, num_units=128)
        self.assertEqual(outputs.get_shape(), [maxlen, 128])

    def test_multihead_attention(self):
        N=64
        Tq=200
        Tk=300
        num_units = 512
        queries = tf.placeholder(shape=[N, Tq, 20], dtype=tf.float32)
        keys = tf.placeholder(shape=[N, Tk, 30], dtype=tf.float32)
        outputs = multihead_attention(queries, keys, num_units=num_units, num_heads=8)
        self.assertEqual(outputs.get_shape(), [N, Tq, num_units])

    def test_feedforward(self):
        inputs = tf.placeholder(shape=[64,200, 512], dtype=tf.float32)
        outputs = feedforward(inputs, scope='forward')
        self.assertEqual(outputs.get_shape(), [64, 200, 512])

if __name__ == '__main__':
    unittest.main()