from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf
import numpy as np
import argparse

def Affine(name_scope,input_tensor,out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
        h = tf.matmul(input_tensor, weights) + biases
        if relu: return tf.nn.relu(h)
        else: return h

def get_batch(bs,all_X,all_sup_Nary_Y,all_sup_Y,d,K):
    inds = np.random.randint(0,d,bs)
    X = np.zeros((bs,d),dtype=np.float32)
    Z = (np.sign(np.random.randn(bs,K))).astype(np.float32)
    Y = np.zeros((bs,K),dtype=np.float32)
    for j,ind in enumerate(inds):
        X[j]=all_X[ind]
        Y[j]=all_sup_Y[ind]
        Z[j,all_sup_Nary_Y[ind]] = 1
    return X,Z,Y
    
def run_training(args):

    all_X = np.eye(args.d,args.d, dtype=np.float32)
    all_sup_Nary_Y = np.random.randint(0,args.k,args.d)
    all_sup_Y = np.zeros((args.d,args.k),dtype=np.float32)
    for j in range(args.d): all_sup_Y[j,all_sup_Nary_Y[j]] = 1
    
    with tf.Graph().as_default():
        session = tf.Session()
        X_placeholder = tf.placeholder(tf.float32, shape=(None,args.d))
        Z_placeholder = tf.placeholder(tf.float32, shape=(None,args.k))
        sup_Y_placeholder = tf.placeholder(tf.float32, shape=(None,args.k))
        
        p1 = Affine('p1', X_placeholder, args.k, relu=False)
        probs = tf.nn.softmax(p1)
        probsZ = tf.reduce_sum(probs*Z_placeholder,1)
        loss_total = tf.reduce_mean(-probsZ)
        if args.Dec:
            probsY = tf.reduce_sum(probs*sup_Y_placeholder,1)
            loss = tf.reduce_mean(-probsY)
        else:
            loss = loss_total
         
        optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5)
        train_op = optimizer.minimize(loss)
        
        session.run(tf.initialize_all_variables())
        for step in xrange(args.num_iters):
            X,Z,Y = get_batch(args.batch_size,all_X,all_sup_Nary_Y,all_sup_Y,args.d,args.k)
            _ = session.run(train_op,feed_dict={X_placeholder:X, 
                                                Z_placeholder: Z,
                                                sup_Y_placeholder: Y})
            if (step % args.print_freq == 0) or step+1 == args.num_iters:
                X,Z,Y = get_batch(500,all_X,all_sup_Nary_Y,all_sup_Y,args.d,args.k)
                loss_ = session.run([loss_total],feed_dict={X_placeholder:X, 
                                                           Z_placeholder: Z,
                                                           sup_Y_placeholder: Y})[0]
                print('Iteration %d loss %.4f'%(step,loss_))

def main(args):
    
    print('Decomposition' if args.Dec else 'E2E')
    run_training(args)

def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", default=1000, type=int, help='input dimension')
    parser.add_argument("--k", default=100, type=int, help='Output distribution dimension')
    parser.add_argument("--Dec", action='store_true', help='Decomposition experiment')
    parser.add_argument("--E2E", action='store_true', help='End to End experiment')
    parser.add_argument("--num_iters", default=2500, type=int, help='number of iterations')
    parser.add_argument("--print_freq", default=100, type=int, help='print frequency')
    parser.add_argument("--batch_size", default=100, type=int, help='batch size')
    args = parser.parse_args()
    if not (args.E2E or args.Dec):
        print('Please choose an experiment.')
        exit(1)
    return args

if __name__ == '__main__':
    main(get_command_line_args())

    