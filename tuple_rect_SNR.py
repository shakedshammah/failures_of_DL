from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import sys
import tensorflow as tf
import numpy as np
import argparse
from read_rect_data import RectTupleData, IM_SIZE, N

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

def Affine(name_scope,input_tensor,out_channels, relu=True, init_sess=None):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
        if init_sess is not None: init_sess.run(tf.initialize_variables([weights,biases]))
        if relu: return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        else: return tf.matmul(input_tensor, weights) + biases


                    

MIN_LEN_TUPLE = 1
MAX_LEN_TUPLE = 4
NUM_ESTIMATES = 1
BATCH_SIZE=100

def run_SNR_estimation(args):
    SNRs = {approach: np.zeros((args.MAX_LEN_TUPLE-args.MIN_LEN_TUPLE+1,args.NUM_ESTIMATES,args.NUM_ESTIMATES,3)) for approach in ['e2e','dec']}
    for e1 in range(args.NUM_ESTIMATES):
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
            session.run(tf.initialize_all_variables()) 
            
            for LEN_TUPLE in range(args.MIN_LEN_TUPLE,args.MAX_LEN_TUPLE+1):
                print('LEN_TUPLE %d'%(LEN_TUPLE))
                data_set = RectTupleData(LEN_TUPLE,args.DATA_PATH)
                for e2 in range(args.NUM_ESTIMATES):
                    p_middle = Affine('p_middle', tf.reshape(tf.sigmoid(score),[-1,LEN_TUPLE]), 100, relu=True, init_sess=session)
                    p_e2e = Affine('p_e2e', p_middle, 1, relu=False, init_sess=session)
                    
                    resh_score = tf.reshape(score,[-1])
                    resh_p_e2e = tf.reshape(p_e2e,[-1])
                    
                    prod_for_dec = tf.slice(tf.reshape(resh_score*Y_sup_placeholder,[-1,LEN_TUPLE]),[0,0],[-1,1])
                    loss = {'e2e': tf.reduce_mean(tf.nn.relu(1-resh_p_e2e*Y_placeholder)), 
                            'dec': tf.reduce_mean(tf.nn.relu(1-prod_for_dec))}
                    
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
                    
                    my_var = [_ for _ in tf.all_variables() if _.name=='score/weights:0']
                    my_var_grad,my_var_grad_norm = {approach: None for approach in loss},{approach: 0 for approach in loss}
                    gav_ops = {approach: optimizer.compute_gradients(loss[approach], my_var) for approach in loss}
                    print('Calculating Gradients')
                    for step in xrange(N//BATCH_SIZE):
                        sys.stderr.write('\r%d/%d'%(step,N//BATCH_SIZE))
                        x,y,y_sup = data_set.next_batch(BATCH_SIZE)
                        fd = {images_placeholder: x, Y_placeholder: y, Y_sup_placeholder: y_sup}
                        gavs_ = {approach: session.run(gav_ops[approach],feed_dict=fd)[0][0] for approach in gav_ops}
                        for approach in SNRs:
                            if my_var_grad[approach] is not None: my_var_grad[approach] += gavs_[approach]
                            else: my_var_grad[approach] = 1*gavs_[approach]
                            my_var_grad_norm[approach] += np.linalg.norm(gavs_[approach])**2
                    sys.stderr.write('\rDONE\n')
                    for approach in SNRs:
                        my_var_grad[approach]/=(N/BATCH_SIZE) 
                        my_var_grad_norm[approach]*=BATCH_SIZE/(N/BATCH_SIZE)
                        SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,0] += np.linalg.norm(my_var_grad[approach])**2
                        SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,1] += my_var_grad_norm[approach]
                        Noise = SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,1] - SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,0]**2
                        SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,2] += SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,e1,e2,0]/Noise
                        print('%d %d LEN_TUP %d %s S %f N %f SNR %f'%(e1,e2,LEN_TUPLE,approach,
                                                                      SNRs[approach][LEN_TUPLE-MIN_LEN_TUPLE,e1,e2,0],
                                                                      SNRs[approach][LEN_TUPLE-MIN_LEN_TUPLE,e1,e2,1],
                                                                      SNRs[approach][LEN_TUPLE-MIN_LEN_TUPLE,e1,e2,2]))
        
    for approach in SNRs:
        SNRs[approach].astype(np.float32).tofile(args.DATA_PATH+'/tuple_rect_SNR_%s_LEN_TUPLES_%d_%d.bin'%(approach,MIN_LEN_TUPLE,MAX_LEN_TUPLE))
        print(approach)
        for LEN_TUPLE in range(args.MIN_LEN_TUPLE,args.MAX_LEN_TUPLE+1):
            print('LEN_TUP %d S %f N %f SNR %f'%(LEN_TUPLE,
                                                 np.mean(SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,:,:,0].flatten()),
                                                 np.mean(SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,:,:,1].flatten()),
                                                 np.mean(SNRs[approach][LEN_TUPLE-args.MIN_LEN_TUPLE,:,:,2].flatten())))
            
def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--MIN_LEN_TUPLE", default=MIN_LEN_TUPLE, type=int, help='MIN_LEN_TUPLE')
    parser.add_argument("--MAX_LEN_TUPLE", default=MAX_LEN_TUPLE, type=int, help='MAX_LEN_TUPLE')
    parser.add_argument("--NUM_ESTIMATES", default=NUM_ESTIMATES, type=int, help='NUM_ESTIMATES')
    parser.add_argument("--DATA_PATH", default='', type=str, help='Data path')
    args = parser.parse_args()
    assert args.DATA_PATH!=''
    return args
    
if __name__ == '__main__':
    run_SNR_estimation(get_command_line_args())

    
