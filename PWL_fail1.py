from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import tensorflow as tf
import numpy as np

import argparse
import os
import glob

import matplotlib.pyplot as plt
from matplotlib import style

def affine(name_scope, input_tensor, out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))), name='weights')
        biases = tf.Variable(tf.zeros([out_channels]), name='biases')
        if relu:
            return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        else:
            return tf.matmul(input_tensor, weights) + biases

class GenericTrainer:
    def __init__(self):
        save_path = '/tmp/PWL' + self.__class__.__name__ + '/'
        self.create_network()
        self.create_loss()
        self._learning_rate = tf.placeholder(tf.float32, shape=[])
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)
        self._init = tf.initialize_all_variables()
        self._save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self._sess = tf.Session()
        self._sess.run(self._init)
        self._saver = tf.train.Saver()
        self._data, self._validation_data = None, None

    def create_network(self):
        raise "not implemented"

    def create_loss(self):
        raise "not implemented"

    def create_data(self, m):
        raise "not implemented"

    def create_data_for_training(self, num_examples, num_validation):
        self._data = self.create_data(num_examples)
        self._validation_data = self.create_data(num_validation)

    def create_dict(self, data, indices=None):
        raise "not implemented"

    def print_status(self, step):
        raise "not implemented"

    def load_snapshot(self, file_name=None):
        snapshot_file = file_name if file_name is not None else max(glob.iglob(self._save_path + "/model*"), key=os.path.getctime)
        self._saver.restore(self._sess, snapshot_file)

    def run_training(self, batch_size, num_examples, num_validation_examples, num_iters, learning_rate):
        self.create_data_for_training(num_examples, num_validation_examples)
        print_interval = num_iters // 10
        for step in xrange(num_iters):
            examples = np.random.randint(num_examples, size=batch_size)
            fd = self.create_dict(self._data, examples)
            fd[self._learning_rate] = learning_rate
            _ = self._sess.run(self._train_op, feed_dict=fd)
            if step % print_interval == 0 and step > 0:
                self.print_status(step)

class FtoKTrainer(GenericTrainer):

    def __init__(self):
        self._n = 100
        self._num_pieces = 3
        self._x = np.array(range(self._n), dtype=np.float32)[:, np.newaxis]
        self._W = np.hstack([np.maximum(0, self._x - i + 1) for i in range(self._n)]).T
        self._invW = np.linalg.inv(self._W)

        GenericTrainer.__init__(self)

    def create_data(self, m):
        pstar = np.zeros((m, self._n), dtype=np.float32)
        all_slopes = [[1, -1, 1], [1, -2, 1], [-2, 1, -1], [1, -1, 2]]
        for i in range(m):
            thetas = 3 + np.sort(np.random.choice(self._n - 6, self._num_pieces, replace=False))
            slopes = all_slopes[np.random.choice(len(all_slopes), 1)[0]]
            a = np.zeros(self._num_pieces, dtype=np.float32)
            s = 0.0
            for j in range(self._num_pieces):
                a[j] = slopes[j] - s
                s = s + a[j]
            pstar[i, thetas] = a
        f = np.dot(pstar, self._W)
        return {'X': f, 'Y': pstar}

    def create_network(self):
        self._X_placeholder = tf.placeholder(tf.float32, shape=(None, self._n))
        self._weights = tf.Variable(tf.zeros([self._n, self._n]), name='weights')
        self._p = tf.matmul(self._X_placeholder, self._weights)

    def create_loss(self):
        self._Y_placeholder = tf.placeholder(tf.float32, shape=(None, self._n))
        self._loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self._p, self._Y_placeholder), reduction_indices=[1]))

    def create_dict(self, data, indices=None):
        if indices is not None:
            return {self._X_placeholder: data['X'][indices], self._Y_placeholder: data['Y'][indices]}
        else:
            return {self._X_placeholder: data['X'], self._Y_placeholder: data['Y']}

    def print_status(self, step):
        validation_loss, invw_ = self._sess.run([self._loss, self._weights],
                                                self.create_dict(self._validation_data))
        print('Step %d: validation loss = %.4f' % (step, validation_loss))
        print('max(abs(invw_-invW)) = %.4f' % np.max(np.abs(invw_ - self._invW)))
        self._saver.save(self._sess, self._save_path + "/model{}".format(step))

    def visualize(self):
        num_examples = 10
        if self._validation_data is None:
            self._validation_data = self.create_data(num_examples)
        p_ = self._sess.run(self._p, feed_dict=self.create_dict(self._validation_data, range(num_examples)))
        for i in range(num_examples):
            style.use('ggplot')
            plt.plot(self._x, self._validation_data['X'][i, :], 'b-', lw=4.1)
            plt.plot(self._x, np.dot(p_[i], self._W), 'r-', lw=2.0)
            plt.show()
            plt.clf()

class FtoKConvTrainer(FtoKTrainer):

    def create_network(self):
        self._X_placeholder = tf.placeholder(tf.float32, shape=(None, self._n))
        self._F_matrix_col0 = tf.slice(self._X_placeholder, [0, 2], [-1, self._n - 2])
        self._F_matrix_col1 = tf.slice(self._X_placeholder, [0, 1], [-1, self._n - 2])
        self._F_matrix_col2 = tf.slice(self._X_placeholder, [0, 0], [-1, self._n - 2])
        self._F_matrix = tf.reshape(
            tf.pack([self._F_matrix_col0, self._F_matrix_col1, self._F_matrix_col2], axis=2), [-1, 3])
        self._weights = tf.Variable(tf.zeros([3, 1]), name='weights')
        self._valid_p = tf.reshape(tf.matmul(self._F_matrix, self._weights), [-1, self._n - 2])
        self._p = tf.pad(self._valid_p, [[0, 0], [2, 0]], "CONSTANT")

    def create_loss(self):
        self._Y_placeholder = tf.placeholder(tf.float32, shape=(None, self._n))
        self._loss = tf.reduce_mean(
            tf.squared_difference(self._p, self._Y_placeholder))

    def create_dict(self, data, indices=None):
        if indices is not None:
            return {self._X_placeholder: data['X'][indices],
                    self._Y_placeholder: data['Y'][indices]}
        else:
            return {self._X_placeholder: data['X'],
                    self._Y_placeholder: data['Y']}

    def print_status(self, step):
        validation_loss, filter_ = self._sess.run(
            [self._loss, self._weights], self.create_dict(self._validation_data))
        print('Step %d: validation loss = %.4f' % (step, validation_loss))
        print('filter = ', filter_.reshape(-1))
        self._saver.save(self._sess, self._save_path + "/model{}".format(step))

    def visualize(self):
        num_examples = 10
        if self._validation_data is None:
            self._validation_data = self.create_data(num_examples)
        p_ = self._sess.run(self._p, feed_dict=self.create_dict(self._validation_data, range(num_examples))).reshape(-1, self._n)
        for i in range(num_examples):
            style.use('ggplot')
            plt.plot(self._x, self._validation_data['X'][i, :], 'b-', lw=4.1)
            plt.plot(self._x, np.dot(p_[i], self._W), 'r-', lw=2.0)
            plt.show()
            plt.clf()

class FtoKConvCondTrainer(FtoKConvTrainer):

    def create_network(self):
        self._np_cond_mat = None
        self._X_placeholder = tf.placeholder(tf.float32, shape=(None, self._n))
        self._F_matrix_col0 = tf.slice(self._X_placeholder, [0, 2], [-1, self._n - 2])
        self._F_matrix_col1 = tf.slice(self._X_placeholder, [0, 1], [-1, self._n - 2])
        self._F_matrix_col2 = tf.slice(self._X_placeholder, [0, 0], [-1, self._n - 2])
        self._F_matrix = tf.reshape(
            tf.pack([self._F_matrix_col0, self._F_matrix_col1, self._F_matrix_col2], axis=2), [-1, 3])
        self._conditioner = tf.placeholder(tf.float32, shape=(3, 3))
        self._Cond_F_matrix = tf.matmul(self._F_matrix, self._conditioner)
        self._weights = tf.Variable(tf.zeros([3, 1]), name='weights')
        self._valid_p = tf.reshape(tf.matmul(self._Cond_F_matrix, self._weights), [-1, self._n - 2])
        self._p = tf.pad(self._valid_p, [[0, 0], [2, 0]], "CONSTANT")

    def create_data_for_training(self, num_examples, num_validation):
        self._data = self.create_data(num_examples)
        self._validation_data = self.create_data(num_validation)
        F = self._sess.run(self._F_matrix, feed_dict={self._X_placeholder: self._data['X']})
        C = np.matmul(F.T, F) / F.shape[0]
        s, V = np.linalg.eigh(C)
        self._np_cond_mat = np.matmul(np.matmul(V, np.diag(s ** (-0.5))), V.T)

    def create_dict(self, data, indices=None):
        assert(self._np_cond_mat is not None)
        if indices is not None:
            return {self._X_placeholder: data['X'][indices], self._conditioner: self._np_cond_mat,
                    self._Y_placeholder: data['Y'][indices]}
        else:
            return {self._X_placeholder: data['X'], self._conditioner: self._np_cond_mat,
                    self._Y_placeholder: data['Y']}

class FAutoEncoderTrainer(GenericTrainer):

    def __init__(self):
        self._n = 100
        self._num_pieces = 3
        self._batchsize = 100
        GenericTrainer.__init__(self)

        self._x = np.array(range(self._n), dtype=np.float32)[:, np.newaxis]

    def create_data(self, m):
        k, n = self._num_pieces, self._n
        x = self._x
        thetas = np.random.uniform(low=0, high=n, size=(m, k))
        coeffs = np.random.uniform(low=-1.0, high=1.0, size=(m, k))
        xx = np.tile(x, (m, 1, k))
        t = np.tile(thetas.reshape(m * k, 1),
                    (1, n)).reshape(m, k, n).transpose(0, 2, 1)
        c = np.tile(coeffs.reshape(m * k, 1),
                    (1, n)).reshape(m, k, n).transpose(0, 2, 1)
        f = np.sum(np.maximum(0, xx - t)*c, axis=2)
        return f.astype(dtype=np.float32)

    def create_network(self):
        self._f_placeholder = tf.placeholder(tf.float32, shape=(self._batchsize, self._n))
        h1 = affine("affine1", self._f_placeholder, 500)
        h2 = affine("affine2", h1, 100)
        self._p = affine("affine3", h2, self._num_pieces*2)

    def create_loss(self):
        h1 = affine("affine4", self._p, 100)
        h2 = affine("affine5", h1, 100)
        self._f = affine("affine6", h2, self._n, relu=False)
        self._loss = tf.reduce_mean(tf.squared_difference(self._f, self._f_placeholder))

    def create_dict(self, data, indices=None):
        if indices is not None:
            return {self._f_placeholder: data[indices, :]}
        else:
            return {self._f_placeholder: data}

    def print_status(self, step):
        num_valid_iters = len(self._validation_data) // self._batchsize
        validation_loss = 0
        for i in range(num_valid_iters):
            examples = [j+i*self._batchsize for j in range(self._batchsize)]
            validation_loss += self._sess.run(self._loss,
                                              feed_dict=self.create_dict(self._validation_data, examples))
        print('Step %d: validation loss = %.4f' % (step, validation_loss/num_valid_iters))
        self._saver.save(self._sess, self._save_path + "/model", global_step=step)

    def visualize(self):
        f = self.create_data(self._batchsize)
        f_ = self._sess.run(self._f,
                            feed_dict={self._f_placeholder: f})
        for i in range(10):
            style.use('ggplot')
            plt.plot(self._x, f[i, :], 'b-', self._x, f_[i, :], 'r-')
            plt.show()

def condition_per_n(start, stop, step):
    c = []
    for n in range(start,stop,step):
        x = np.array(range(n), dtype=np.float32)[:, np.newaxis]
        W = np.hstack([np.maximum(0, x - i + 1) for i in range(n)])
        s = np.linalg.svd(W, compute_uv=False)
        c.append(s[0]/s[-1])
    (c - np.array([v * v for v in range(start, stop, step)])) / c

def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--FtoK", action='store_true', help='FtoK')
    parser.add_argument("--FtoKConv", action='store_true', help='FtoKConv')
    parser.add_argument("--FtoKConvCond", action='store_true', help='FtoKConv')
    parser.add_argument("--FAutoEncoder", action='store_true', help='AutoEncoder')
    parser.add_argument("--snapshot_file", default='', help='Restore snapshot')
    parser.add_argument("--batch_size", default=100, type=int, help='batch size')
    parser.add_argument("--number_of_iterations", default=1000, type=int, help='batch size')
    parser.add_argument("--learning_rate", default=0.01, type=float, help='batch size')
    args = parser.parse_args()
    return args

def main(args):
    if args.FtoK:
        trainer = FtoKTrainer()
        if args.snapshot_file == '':
            trainer.run_training(100, 10000, 1000, 50000, 0.0001)
        else:
            trainer.load_snapshot(args.snapshot_file)
        trainer.visualize()
    if args.FtoKConv:
        trainer = FtoKConvTrainer()
        trainer.run_training(100, 10000, 1000, 50000, 0.05)
        trainer.visualize()
    if args.FtoKConvCond:
        trainer = FtoKConvCondTrainer()
        trainer.run_training(args.batch_size, 10000, 1000, args.number_of_iterations, args.learning_rate)
        trainer.visualize()
    if args.FAutoEncoder:
        trainer = FAutoEncoderTrainer()
        trainer.run_training(trainer._batchsize, 10000, 1000, 10000, 0.001)
        trainer.visualize()

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = get_command_line_args()
    main(args)

