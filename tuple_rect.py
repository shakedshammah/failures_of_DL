from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import tensorflow as tf
import numpy as np
import argparse
from read_rect_data import RectTupleData, IM_SIZE

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def conv_maxpool(name_scope,input_tensor,out_channels,kernel_size):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    new_shape = [_ for _ in input_shape]+[1]*(4-len(input_shape))
    new_shape = [_ if _ is not None else -1 for _ in new_shape]
    inp_resh = tf.reshape(input_tensor, new_shape)
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([kernel_size,kernel_size,new_shape[-1], out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
        h_conv = conv2d(inp_resh, weights) + biases
        return max_pool_2x2(h_conv)

def Affine(name_scope,input_tensor,out_channels, relu=True):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
#         initializer = tf.initialize_variables([weights,biases])
        if relu: return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)#,initializer
        else: return tf.matmul(input_tensor, weights) + biases#,initializer

                    
    
BATCH_SIZE = 200
MAX_STEPS = 20000
MIN_LEN_TUPLE = 1
MAX_LEN_TUPLE = 4
PRINT_ME = 100

def run_SNR_estimation(args):
    for LEN_TUPLE in range(args.MIN_LEN_TUPLE,args.MAX_LEN_TUPLE+1):
        print('LEN_TUPLE %d'%(LEN_TUPLE))
        data_set = RectTupleData(LEN_TUPLE,args.DATA_PATH)
        data_set_test = RectTupleData(LEN_TUPLE,args.DATA_PATH,n=BATCH_SIZE*10)
        with tf.Graph().as_default():
            session = tf.Session()
            images_placeholder = tf.placeholder(tf.float32, shape=(None,IM_SIZE,IM_SIZE))
            Y_sup_placeholder = tf.placeholder(tf.float32, shape=(None))
            Y_placeholder = tf.placeholder(tf.float32, shape=(None))
            
            p1 = conv_maxpool('p1', images_placeholder, 16, 5)
            p2 = conv_maxpool('p2', p1, 16, 5)
            p2_flat = tf.reshape(p2,[-1,int(np.prod(p2._shape[1:]))])
            p3 = Affine('p3', p2_flat, 50, relu=True)
            p4 = Affine('p4', p3, 100, relu=True)
            score = Affine('score', p4, 1, relu=False)
            
            p_middle = Affine('p_middle', tf.reshape(tf.sigmoid(score),[-1,LEN_TUPLE]), 100, relu=True)
            p_e2e = Affine('p_e2e', p_middle, 1, relu=False)
            
            resh_score = tf.reshape(score,[-1])
            resh_p_e2e = tf.reshape(p_e2e,[-1])
            
            loss = {'e2e': tf.reduce_mean(tf.nn.relu(1-resh_p_e2e*Y_placeholder)), 
                    'dec': tf.reduce_mean(tf.nn.relu(1-resh_score*Y_sup_placeholder))}
            
            zero_one_dec = tf.reduce_mean(tf.cast(resh_score*Y_sup_placeholder>0,tf.float32))
            zero_one_e2e = tf.reduce_mean(tf.cast(resh_p_e2e*Y_placeholder>0,tf.float32))
            
            optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=0.9, beta2=0.99, epsilon=1e-5)
            approaches_to_min = ['e2e']
            if args.dec: approaches_to_min.append('dec')
            all_loss = loss[approaches_to_min[0]]
            for atm in approaches_to_min[1:]: all_loss += loss[atm]
            
            train_op = optimizer.minimize(all_loss)
            log_ = []
            session.run(tf.initialize_all_variables())
            for step in xrange(args.MAX_STEPS+1):
                x,y,y_sup = data_set.next_batch(args.BATCH_SIZE)
                
                _ = session.run(train_op,feed_dict={images_placeholder: x, 
                                                    Y_placeholder: y, 
                                                    Y_sup_placeholder: y_sup})
                if (step%PRINT_ME==0):
                    x,y,y_sup = data_set_test.next_batch(BATCH_SIZE*10)
                    loss_e2e_,loss_dec_,zo_dec_,zo_e2e_ = session.run([loss['e2e'],loss['dec'],zero_one_dec,zero_one_e2e],feed_dict={images_placeholder: x, 
                                                        Y_placeholder: y, 
                                                        Y_sup_placeholder: y_sup})
                    print('TEST iter %d lossE2E %f lossDEC %f AccuracyDEC %f AccuracyE2E %f'%(step,loss_e2e_,loss_dec_,zo_dec_,zo_e2e_))
                    log_.append([loss_e2e_,loss_dec_,zo_dec_,zo_e2e_])
                    x,y,y_sup = data_set.next_batch(BATCH_SIZE*10)
                    
                    loss_e2e_,loss_dec_,zo_dec_,zo_e2e_ = session.run([loss['e2e'],loss['dec'],zero_one_dec,zero_one_e2e],feed_dict={images_placeholder: x, 
                                                        Y_placeholder: y, 
                                                        Y_sup_placeholder: y_sup})
                    print('TRAIN iter %d lossE2E %f lossDEC %f AccuracyDEC %f AccuracyE2E %f'%(step,loss_e2e_,loss_dec_,zo_dec_,zo_e2e_))
            np.array(log_).astype(np.float32).tofile(args.DATA_PATH+'/tuple_rect_log_%d_%s.bin'%
                                                     (LEN_TUPLE,'_'.join(approaches_to_min)))            


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MIN_LEN_TUPLE", default=MIN_LEN_TUPLE, type=int, help='MIN_LEN_TUPLE')
    parser.add_argument("--MAX_LEN_TUPLE", default=MAX_LEN_TUPLE, type=int, help='MAX_LEN_TUPLE')
    parser.add_argument("--BATCH_SIZE", default=BATCH_SIZE, type=int, help='BATCH_SIZE')
    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--DATA_PATH", default='', type=str, help='Data path')
    parser.add_argument("--dec", action='store_true', help='decompose')
    parser.add_argument("--MAX_STEPS", default=MAX_STEPS, type=int,  help='MAX_STEPS')
    args = parser.parse_args()
    assert args.DATA_PATH!=''
    return args
if __name__ == '__main__':
    run_SNR_estimation(get_command_line_args())

    
