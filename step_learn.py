#!/usr/bin/python

import numpy as np
import tensorflow as tf
import sys
import math
import pdb
import argparse
import matplotlib.pyplot as plt

N = 100
# This is the size of the floor discretization
floor_step = 0.1
factor_W = 1
BATCH_SIZE = 256
PRINT_FREQUENCY = 200
L = 10.0
def sort_by_p(X,p): return [_[0] for _ in sorted(zip(X,p),key=lambda x: x[1])]

def my_floor(p,Z):
    Y = np.zeros_like(p)
    for i in range(len(p)):
        Y[i] = np.max(Z - 9999999.0*((p[i]<Z).astype(np.float32)))
    return Y[:,np.newaxis]

# a simple function for implementing an affine tranformation
def Affine(name_scope,input_tensor,out_channels, relu=False):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(name_scope):
        weights = tf.Variable(
            tf.truncated_normal([input_channels, out_channels],
                                stddev=1.0 / math.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([out_channels]),name='biases')
        if relu: return tf.nn.relu(tf.matmul(input_tensor, weights) + biases),weights,biases
        else: return tf.matmul(input_tensor, weights) + biases,weights,biases

class GeneralFloorLearn():
    def __init__(self, W_star, b_star, Z):
        self._W_star, self._b_star, self._Z = W_star, b_star, Z
    def my_show(self):
        plt.title(self.__class__.__name__)
        plt.show()
        plt.clf()
    def my_plot(self,session,fd,y,Y,p,p1,Z_placeholder,additional_placeholders):
        y_val = session.run([y],feed_dict=fd)[0]
        Y = [_[0] for _ in sorted(zip(Y,p),key=lambda x: x[1])]
        y_val = [_[0] for _ in sorted(zip(y_val,p),key=lambda x: x[1])]
        p = sorted(p)
        plt.plot(p,y_val,'r');plt.plot(p,Y,'g--');
        self.my_show()
    def get_lr(self): return 0.001
    def max_steps(self): return 10000
    def additional_placeholders(self): return dict()
    def update_additional_placeholders(self,additional_placeholders,X,p,Y): pass
    def get_architecture(self, X_placeholder, additional_placeholders): raise 'Not Implemented'
    def get_y(self,p1,additional_placeholders): raise 'Not Implemented'
    def get_loss(self,p1,y,Y_placeholder, additional_placeholders):
        return tf.reduce_mean(tf.squared_difference(y,Y_placeholder))
    def create_data(self, batch_size,X_placeholder,Y_placeholder,Z_placeholder,additional_placeholders):
        X = np.random.randn(batch_size,N)/N
        p = np.dot(X,self._W_star.T) + self._b_star
        Y = my_floor(p,self._Z)
        Z = np.repeat(self._Z[np.newaxis,:], repeats=X.shape[0], axis=0)
        fd = {X_placeholder: X, Y_placeholder: Y, Z_placeholder: Z}
        self.update_additional_placeholders(additional_placeholders,X,p,Y)
        fd.update(dict(additional_placeholders.values()))
        fd = {k:v for k,v in fd.iteritems() if v is not None}
        return p,Y,fd
    
    def train_me(self):
        print('Training %s'%(self.__class__.__name__))
        with tf.Graph().as_default():
            session = tf.Session()
            X_placeholder = tf.placeholder(tf.float32, shape=(None,N))
            Y_placeholder = tf.placeholder(tf.float32, shape=(None,1))
            Z_placeholder = tf.placeholder(tf.float32, shape=(None,len(self._Z)))
            additional_placeholders = self.additional_placeholders()
            p1 = self.get_architecture(X_placeholder, additional_placeholders)
            y = self.get_y(p1,Z_placeholder,additional_placeholders)
            loss = self.get_loss(p1,y,Y_placeholder, additional_placeholders)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.get_lr(), beta1=0.9, beta2=0.99, epsilon=1e-5)
            train_op = optimizer.minimize(loss)        
            session.run(tf.initialize_all_variables())
            for step in xrange(self.max_steps()):
                # New batch
                p,Y,fd = self.create_data(BATCH_SIZE,X_placeholder,Y_placeholder,Z_placeholder,additional_placeholders)
                _ = session.run([train_op],feed_dict=fd)
                
                if (step > 0 and step % PRINT_FREQUENCY == 0) or step+1 == self.max_steps():
                    p,Y,fd = self.create_data(BATCH_SIZE,X_placeholder,Y_placeholder,Z_placeholder,additional_placeholders)
                    loss_val = session.run(loss,feed_dict=fd)
                    print 'step %d, loss = %f'%(step,loss_val)
                    if (step+1 == self.max_steps()):
                        p,Y,fd = self.create_data(BATCH_SIZE*10,X_placeholder,Y_placeholder,Z_placeholder,additional_placeholders)
                        self.my_plot(session,fd,y,Y,p,p1,Z_placeholder,additional_placeholders)
                        
class IsotronFloorLearn(GeneralFloorLearn):
    def get_architecture(self, X_placeholder, additional_placeholders):
        p1,_,_ = Affine('p1', X_placeholder, 1)
        return p1
    def max_steps(self):
        return 5000
    def get_y(self,p1,Z_placeholder,additional_placeholders):
        p1_tile = tf.tile(p1, tf.pack([1, len(self._Z)]))
        geq_Z = tf.cast(p1_tile>=Z_placeholder,tf.float32)
        geq_Z_0 = tf.slice(geq_Z,[0,0],[-1,len(self._Z)-1])
        geq_Z_1 = tf.slice(geq_Z,[0,1],[-1,len(self._Z)-1])
        geq_diff = -geq_Z_1 + geq_Z_0 
        # Now there's 1 at only a single coordinate, corresponding to the largest one which we are geq.
        # There's all zeros if we are geq than the last, but this is unlikely
        slice_Z = tf.slice(Z_placeholder, [0,0],[1,len(self._Z)-1])
        curr_floor = tf.reshape(tf.matmul(slice_Z,tf.transpose(geq_diff,[1,0])),[-1,1])
        return p1 + tf.stop_gradient(-p1 + curr_floor)
   
class EndToEndFloorLearn(GeneralFloorLearn):
    def get_architecture(self, X_placeholder, additional_placeholders):
        p1,_,_ = Affine('p1', X_placeholder, 100, relu=True)
        p2,_,_ = Affine('p2', p1, 100, relu=True)
        p3,_,_ = Affine('p3', p2, 100, relu=True)
        p4,_,_ = Affine('p4', p3, 1)
        return p4
    def get_y(self,p1,Z_placeholder,additional_placeholders):
        return p1
    
class DifferentiableApproximationFloorLearn(GeneralFloorLearn):
    def get_lr(self): return 0.01
    def additional_placeholders(self): 
        return {'correct_p1_placeholder': [tf.placeholder(tf.float32, shape=(None,1)),None]}
    def get_architecture(self, X_placeholder, additional_placeholders):
        p1,_,_ = Affine('p1', X_placeholder, 1)
        return p1
    def get_y(self,p1,Z_placeholder,additional_placeholders):
        p1_tile = tf.tile(p1, tf.pack([1, len(self._Z)-1]))
        
        first_Z = tf.slice(Z_placeholder,[0,0],[-1,1])
        slice_Z_0 = tf.slice(Z_placeholder, [0,0],[-1,len(self._Z)-1])
        slice_Z_1 = tf.slice(Z_placeholder, [0,1],[-1,len(self._Z)-1])
        diff_Z = slice_Z_1 - slice_Z_0
        middle_slopes = slice_Z_1 - diff_Z/(2*L)
        just_const = 5
        sigmomiddle = tf.nn.sigmoid(just_const*(p1_tile-middle_slopes)/(diff_Z/L))
        slice_diff_Z = tf.slice(diff_Z, [0,0],[1,-1])
        return first_Z + tf.reshape(tf.matmul(slice_diff_Z,tf.transpose(sigmomiddle,[1,0])),[-1,1])
    
    def my_plot(self,session,fd,y,Y,p,p1,Z_placeholder,additional_placeholders):
        y_val,p1_val = session.run([y,p1],feed_dict=fd)
        y_val_real = my_floor(p1_val.flatten(),self._Z)
        correct_p1 = additional_placeholders['correct_p1_placeholder'][0]
        my_y = self.get_y(correct_p1,Z_placeholder,additional_placeholders)
        my_y_val = session.run(my_y,feed_dict={Z_placeholder: np.repeat(self._Z[np.newaxis,:], repeats=p1_val.shape[0], axis=0),
                                               additional_placeholders['correct_p1_placeholder'][0]: p[:,np.newaxis]})
        Y = sort_by_p(Y,p)
        y_val = sort_by_p(y_val,p)
        my_y_val = sort_by_p(my_y_val,p)
        y_val_real = sort_by_p(y_val_real,p)
        p = sorted(p)
        plt.plot(p,Y,'g--');plt.plot(p,y_val,'r');plt.plot(p,my_y_val,'m--');plt.plot(p,y_val_real,'b')
        self.my_show()
    
class MCFloorLearn(GeneralFloorLearn):
    def additional_placeholders(self): 
        return {'label_placeholder': [tf.placeholder(tf.int32, shape=(None,1)),None]}
    def update_additional_placeholders(self,additional_placeholders,X,p,Y):
        # put into classes by Y
        classes = np.zeros((Y.size,1)).astype(np.int32)
        for i in range(classes.size):
            classes[i] = [_ for _ in enumerate(self._Z) if _[1]<=Y[i]][-1][0]
        additional_placeholders['label_placeholder'][1] = classes
    def get_architecture(self, X_placeholder, additional_placeholders):
        p1,_,_ = Affine('p1', X_placeholder, 100, relu=True)
        p2,_,_ = Affine('p2', p1, 100, relu=True)
        p3,_,_ = Affine('p3', p2, 100)
        return p3
    def get_y(self,p1,Z_placeholder,additional_placeholders): return None
    def get_loss(self,p1,y,Y_placeholder, additional_placeholders):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(p1, 
                                                                             tf.reshape(additional_placeholders['label_placeholder'][0],[-1])))
    def my_plot(self,session,fd,y,Y,p,p1,Z_placeholder,additional_placeholders):
        max_class = tf.arg_max(p1, dimension=1)
        max_class_val = (session.run(max_class,feed_dict=fd)).astype(np.int32)
        y_val = self._Z[max_class_val]
        Y = sort_by_p(Y,p)
        y_val = sort_by_p(y_val,p)
        p = sorted(p)
        plt.plot(p,Y,'g--',p,y_val,'r')
        self.my_show()
                                    
def main(args):
    W_star = factor_W*np.random.randn(N)
    b_star = np.random.randn()
    Z = b_star + np.concatenate((np.r_[-100:100:5],np.array([-5,-4,-3,-2,-1,-0.2,0,0.1,0.15,0.175,1,2,3,4,5])))
    Z = np.array(list(set([_ for _ in Z])))
    Z.sort()
    
    if args.all or args.non_flat_approximation: 
        DifferentiableApproximationFloorLearn(W_star, b_star, Z).train_me()
    if args.all or args.e2e: 
        EndToEndFloorLearn(W_star, b_star, Z).train_me()
    if args.all or args.mc: 
        MCFloorLearn(W_star, b_star, Z).train_me()
    if args.all or args.forward_only:
        IsotronFloorLearn(W_star, b_star, Z).train_me()
        
def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action='store_true', help='Run all experiments')
    parser.add_argument("--non_flat_approximation", action='store_true', help='Nonflat approximation experiment')
    parser.add_argument("--e2e", action='store_true', help='end to end experiment')
    parser.add_argument("--mc", action='store_true', help='MC experiment')
    parser.add_argument("--forward_only", action='store_true', help='forward only experiment')
    args = parser.parse_args()
    if not any([args.all,args.non_flat_approximation,args.e2e,args.mc,args.forward_only]):
        print('Please choose an experiment.')
        exit(1)
    return args
    
if __name__ == '__main__':
    main(get_command_line_args())