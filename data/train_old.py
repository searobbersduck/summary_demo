# !/usr/bin/env python3

import tensorflow as tf
import numpy as np

import os
import sys
sys.path.append('../')
sys.path.append('../bert')
sys.path.append('../recommendrecord')
sys.path.append('./')

from model import *
from data import *


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
    return session

def train():
    opts = parse_args()
    filepattern = opts.filepattern
    vocab_file = './xxx.txt'
    extractor = ResumeTxtExtractor()
    vocab = Vocabulary('./summary_data/xxx.txt')
    ds = ResumeSummaryDataset('./summary_data/seg/file_*', vocab, extractor)
    opts.vocab_size=vocab.size
    # wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
    # ds = ResumeSummaryDataset(filepattern, vocab, extractor)
    ckpt_dir = opts.ckpt_dir
    best_acc = 0
    with tf.Graph().as_default():
        with tf.device('/device:GPU:1'):
            wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
            global_step = tf.train.get_or_create_global_step()
            model = SummaryModel(wordsEmbeddings, opts, True, global_step)
            loss_op = model.loss
            train_op = model.train_op
            acc_op = model.acc
            saver = tf.train.Saver()
            X = ds.iter_batches(opts.batch_size,
                                opts.resume_max_line,
                                opts.resume_max_tokens_per_line,
                                opts.record_max_length)
            sv = tf.train.Supervisor(logdir=ckpt_dir)
            with sv.managed_session(master='',
                                    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
                try:
                    while not sv.should_stop():
                        Y = next(X)
                        # resume_tensor = tf.convert_to_tensor(Y[1], dtype=tf.int32)
                        # summary_tensor = tf.convert_to_tensor(Y[0], dtype=tf.int32)
                        print('xxxxxxxx:')
                        print(Y[1].shape)
                        print('xxxxxxxx:')
                        print(Y[0].shape)
                        train_val, acc_val, loss_val, global_step_val = sess.run([
                            train_op, acc_op, loss_op, global_step
                        ],
                            feed_dict={model.resume_tokens:Y[1], model.summary_tokens:Y[0], model.learning_rate:1e-4})
                        print('loss:{}\tacc:{}'.format(loss_val, acc_val))
                        if acc_val > best_acc:
                            best_acc = acc_val
                            print('Current best accuracy is: {}'.format(best_acc))
                            sv.saver.save(sess, os.path.join(ckpt_dir, 'best_model'))
                except Exception as e:
                    print(e)
                    sv.saver.save(sess, os.path.join(ckpt_dir, 'final_model'))



if __name__ == '__main__':
    train()

