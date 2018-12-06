# !/usr/bin/env python3

import tensorflow as tf

def main(unused_args):
    def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            # pylint: disable=W0212
            session = session._sess
        return session

    with tf.Graph().as_default():
        a=tf.get_variable('a', shape=[10,3])
        b=tf.get_variable('b', shape=[3,4])
        c=tf.matmul(a,b)
        d=tf.get_variable('d', shape=[10,4])
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.GradientDescentOptimizer(1e-4)
        loss = tf.convert_to_tensor(tf.reduce_sum(tf.reduce_sum(c-d, axis=-1)))
        loss = tf.abs(loss)
        grads = opt.compute_gradients(loss)
        min_loss = 0.01
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        saver = tf.train.Saver()
        with tf.train.MonitoredTrainingSession(checkpoint_dir='./test',
                                               hooks=[tf.train.StopAtStepHook(last_step=25000), tf.train.NanTensorHook(loss)]) as sess:
            while not sess.should_stop():
                loss1, op1, global_step1 = sess.run([loss, apply_gradient_op, global_step])
                print(loss1)
                if loss1 < min_loss:
                    min_loss = loss1
                    saver.save(get_session(sess), './test/best_model')
                    print('save best model')


if __name__ == '__main__':
    tf.app.run()