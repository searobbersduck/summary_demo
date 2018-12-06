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
import math

global_beam_search_top_k = 4
global_beam_search_w = 50


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def beam_search_decoder(preds, all_candidates_in, i_pos, k, w):
    all_candidates_out = list()
    for i in range(len(all_candidates_in)):
        seq, score = all_candidates_in[i]
        next_prob = preds[i][i_pos]
        dict_prob = {}
        print('1')
        # next_prob = softmax(np.array(next_prob))
        for ii,prob in enumerate(list(next_prob)):
            dict_prob[ii] = prob
        print('2')
        next_prob = sorted(dict_prob.items(), key=lambda x: x[1], reverse=True)[:w]
        next_prob_arr = np.array([i[1] for i in next_prob])
        next_prob_arr = softmax(next_prob_arr)
        for j in range(len(next_prob)):
            # candidate = [seq + [next_prob[j][0]], score * -math.log(next_prob[j][1])]
            # candidate = [seq + [next_prob[j][0]], score * -math.log(next_prob_arr[j])]
            candidate = [seq + [next_prob[j][0]], score + next_prob_arr[j]]
            all_candidates_out.append(candidate)
    ordered = sorted(all_candidates_out, key=lambda tup: tup[1])
    return ordered[:k]


def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
    return session

# def eval():
#     opts = parse_args()
#     filepattern = opts.filepattern
#     vocab_file = './xxx.txt'
#     extractor = ResumeTxtExtractor()
#     vocab = Vocabulary('./summary_data/xxx.txt')
#     ds = ResumeSummaryDataset('./summary_data/seg/file_*', vocab, extractor)
#     opts.vocab_size = vocab.size
#     # wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
#     # ds = ResumeSummaryDataset(filepattern, vocab, extractor)
#     ckpt_dir = opts.ckpt_dir
#     best_acc = 0
#     with tf.Graph().as_default():
#         with tf.device('/device:GPU:1'):
#             wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
#             global_step = tf.train.get_or_create_global_step()
#             model = SummaryModel(wordsEmbeddings, opts, True, global_step)
#             loss_op = model.loss
#             train_op = model.train_op
#             acc_op = model.acc
#             saver = tf.train.Saver()
#             X = ds.iter_batches(opts.batch_size,
#                                 opts.resume_max_line,
#                                 opts.resume_max_tokens_per_line,
#                                 opts.record_max_length)
#             with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir,
#                                                    hooks=[tf.train.StopAtStepHook(last_step=opts.iter_num)],
#                                                    config=tf.ConfigProto(
#                                                        allow_soft_placement=True, log_device_placement=True)
#                                                    ) as sess:
#                 try:
#                     while not sess.should_stop():
#                         Y = next(X)
#                         # resume_tensor = tf.convert_to_tensor(Y[1], dtype=tf.int32)
#                         # summary_tensor = tf.convert_to_tensor(Y[0], dtype=tf.int32)
#                         target_len = opts.record_max_length
#                         preds = np.zeros([opts.batch_size, target_len], dtype=np.int32)
#                         print('1')
#                         for i in range(target_len): # target_len
#                             _preds, loss_val, acc_val = \
#                                 sess.run([model.preds, loss_op, acc_op], feed_dict={
#                                     model.resume_tokens:Y[1],
#                                     model.summary_tokens:preds,
#                                 })
#                             _preds = np.reshape(_preds, preds.shape)
#                             preds[:, i] = _preds[:, i]
#                         print('2')
#                         example_pred = preds[0]
#                         print(example_pred)
#                         print(vocab.decode(example_pred))
#                         print(vocab.decode(Y[0][0]))
#                         print('loss:{}\tacc:{}'.format(loss_val, acc_val))
#                 except Exception as e:
#                     print(e)
#                     # saver.save(get_session(sess), os.path.join(ckpt_dir, 'final_model'))


def eval():
    opts = parse_args()
    filepattern = opts.filepattern
    vocab_file = './xxx.txt'
    extractor = ResumeTxtExtractor()
    vocab = Vocabulary('./summary_data/v160k_big_string.txt')
    ds = ResumeSummaryDataset('./summary_data/seg/file_*', vocab, extractor)
    opts.vocab_size = vocab.size
    # wordsEmbeddings = tf.Variable(vocab.emb, dtype=tf.float32)
    # ds = ResumeSummaryDataset(filepattern, vocab, extractor)
    ckpt_dir = opts.ckpt_dir
    best_acc = 0
    with tf.Graph().as_default():
        with tf.device('/device:GPU:2'):
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
            with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir,
                                                   hooks=[tf.train.StopAtStepHook(last_step=opts.iter_num)],
                                                   config=tf.ConfigProto(
                                                       allow_soft_placement=True, log_device_placement=True)
                                                   ) as sess:
                try:
                    while not sess.should_stop():
                        Y = next(X)
                        # resume_tensor = tf.convert_to_tensor(Y[1], dtype=tf.int32)
                        # summary_tensor = tf.convert_to_tensor(Y[0], dtype=tf.int32)
                        target_len = opts.record_max_length
                        preds = np.zeros([opts.batch_size, target_len], dtype=np.int32)
                        resume_tokens = np.zeros(Y[1].shape)
                        for ii in range(opts.batch_size):
                            resume_tokens[ii] = Y[1][ii]
                            preds[ii][0] = 1

                        all_candidates = [[[1], 1.0], [[1], 1.0], [[1], 1.0], [[1], 1.0]]
                        for i in range(1, 20): # target_len
                            _preds, loss_val, acc_val, _logits = \
                                sess.run([model.preds, loss_op, acc_op, model.logits], feed_dict={
                                    model.resume_tokens:Y[1],
                                    model.summary_tokens:preds,
                                })
                            # _preds = np.reshape(_preds, preds.shape)
                            print(_logits.shape)
                            _logits = np.reshape(_logits, [preds.shape[0], preds.shape[1], -1])
                            all_candidates = beam_search_decoder(_logits, all_candidates, i,
                                                                 global_beam_search_top_k,
                                                                 global_beam_search_w)
                            # preds[:, i] = _preds[:, i]
                            for j, can in enumerate(all_candidates):
                                seq, score = can
                                print('score:\t', score)
                                print(seq)
                                preds[j][1:i+1] = seq[1:i+1]
                            print('4')
                        print('5')
                        example_pred = preds[0]
                        print(example_pred)
                        print('\n')
                        print('====>predict: ')
                        print(vocab.decode(example_pred))
                        print('====>predict: ')
                        print(vocab.decode(preds[1]))
                        print('====>predict: ')
                        print(vocab.decode(preds[2]))
                        print('====>predict: ')
                        print(vocab.decode(preds[3]))
                        print('\n')
                        print('====>ground truth: ')
                        print(vocab.decode(Y[0][0]))
                        print('loss:{}\tacc:{}'.format(loss_val, acc_val))
                except Exception as e:
                    print(e)
                    # saver.save(get_session(sess), os.path.join(ckpt_dir, 'final_model'))


if __name__ == '__main__':
    eval()
