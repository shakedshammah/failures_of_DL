from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import sys
import tensorflow as tf
import numpy as np
import argparse
import os

def Affine(name_scope,input_tensor,out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(tf.truncated_normal([input_channels, out_channels],
                                                  stddev=1.0 / math.sqrt(float(input_channels))),
                              name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
        h = tf.matmul(input_tensor, weights) + biases
        if relu: return tf.nn.relu(h)
        else: return h

def get_batch(bs,len_subset,N):
    X = 1.0-2*(np.random.randn(bs,N) < 0).astype(np.float32)
    relevant_X = X[:,:len_subset] # Assume that v^* has all its 1's at the prefix
    if len_subset==1: relevant_X = relevant_X[:,np.newaxis]
    Y = np.prod(relevant_X, axis=1)[:,np.newaxis]
    return X,Y

def run_training(args):
    print('Parity d=%d'%(args.d))
    len_subset = np.sum(np.random.randn(args.d)>0) if args.subset_size < 0 else args.subset_size
    with tf.Graph().as_default():
        session = tf.Session()
        # These will be inserted as single images
        X_placeholder = tf.placeholder(tf.float32, shape=(None,args.d))
        Y_placeholder = tf.placeholder(tf.float32, shape=(None,1))
        
        p1 = Affine('p1', X_placeholder, 10*args.d, relu=True)
        score = Affine('score', p1, 1, relu=False)
        # Hinge loss
        loss = tf.reduce_mean(tf.nn.relu(1.0 - Y_placeholder*score))
        
        accuracy = tf.reduce_mean(tf.cast(Y_placeholder*score>0,tf.float32))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)
        
        session.run(tf.initialize_all_variables())
        for step in xrange(args.num_iters):
            X,Y = get_batch(args.batch_size,len_subset,args.d)
            _ = session.run(train_op,feed_dict={X_placeholder: X, Y_placeholder: Y})
            if (step % args.print_freq == 0) or step+1 == args.num_iters:
                X,Y = get_batch(500,len_subset,args.d)
                fd = {X_placeholder: X, Y_placeholder: Y}
                print('\nIteration %d' % (step))
                loss_,accuracy_ = session.run([loss,accuracy],feed_dict=fd)
                print('Iteration %d loss %.4f accuracy %.4f'%(step,loss_,accuracy_))
                
def main(args):
    run_training(args)

def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=100, type=int, help='input dimension')
    parser.add_argument("--num_iters", default=50000, type=int, help='number of iterations')
    parser.add_argument("--print_freq", default=100, type=int, help='print frequency')
    parser.add_argument("--learning_rate", default=0.01, type=float, help='learning rate')
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--subset_size", default=-1, type=int, help='size of subset for parity (explicitly, \
                                                                        sum(v^*==1), see paper section 2.1 for notation). \
                                                                        Negative value will result in random.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(get_command_line_args())

    