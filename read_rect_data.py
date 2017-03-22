from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from itertools import product

IM_SIZE=28
N=500000

class RectTupleData():
    def __init__(self, len_tuple,DATA_PATH,n=N):
        self._len_tuple = len_tuple
        self._cur_ind = 0
        self._N = n
        try:
            self._fid_X = open(DATA_PATH+'/X_%d_%d.bin'%(n,len_tuple),'rb')
            self._fid_Y = open(DATA_PATH+'/Y_%d_%d.bin'%(n,len_tuple),'rb')
            self._fid_Y_sup = open(DATA_PATH+'/Y_sup_%d_%d.bin'%(n,len_tuple),'rb')
        except:
            fid_X = open(DATA_PATH+'/X_%d_%d.bin'%(n,len_tuple),'wb')
            fid_Y = open(DATA_PATH+'/Y_%d_%d.bin'%(n,len_tuple),'wb')
            fid_Y_sup = open(DATA_PATH+'/Y_sup_%d_%d.bin'%(n,len_tuple),'wb')
            self._X = np.zeros((len_tuple,IM_SIZE,IM_SIZE),dtype=np.float32)
            self._Y = 0
            self._Y_sup = np.zeros(len_tuple,dtype=np.int32)
            for i in range(n):
                self._X *=0; self._Y=0; self._Y_sup *=0
                sys.stderr.write('\r%d/%d'%(i,n))
                for k in range(len_tuple):                    
                    theta = np.random.uniform(0,np.pi)
                    unit_vec = np.array([np.cos(theta),np.sin(theta)])
                    self._Y_sup[k] = 1.0 if theta<np.pi/2 else -1.0 
                    self._Y += 1.0 if theta<np.pi/2 else 0.0 # density of theta<pi/2
                    # generate the image
                    center = IM_SIZE*np.random.uniform(0,1,size=2)
                    length = np.random.uniform(5,IM_SIZE-5)
                    for l in range(int(length//2)):
                        for sgn in [-1,1]:
                            pt = (center + sgn*unit_vec*l).astype(np.int32)
                            for dx,dy in product([0],repeat=2):#product([-1,0,1],repeat=2):
                                ptpt=pt+np.array([dx,dy])
                                if all((0<=ptpt)*(ptpt<IM_SIZE)):
                                    self._X[k,ptpt[0],ptpt[1]] = 1
                self._Y = (-1+2*np.mod(self._Y,2)).astype(np.float32)
                self._X.tofile(fid_X); fid_X.flush()
                self._Y.tofile(fid_Y); fid_Y.flush()
                self._Y_sup.tofile(fid_Y_sup);fid_Y_sup.flush()
            fid_X.close();fid_Y.close();fid_Y_sup.close(); 
            self._fid_X = open(DATA_PATH+'/X_%d_%d.bin'%(n,len_tuple),'rb')
            self._fid_Y = open(DATA_PATH+'/Y_%d_%d.bin'%(n,len_tuple),'rb')
            self._fid_Y_sup = open(DATA_PATH+'/Y_sup_%d_%d.bin'%(n,len_tuple),'rb')

    def next_batch(self, BATCH_SIZE):
        
        self._X = np.frombuffer(self._fid_X.read(4*BATCH_SIZE*self._len_tuple*IM_SIZE*IM_SIZE),dtype=np.float32)
        self._Y = np.frombuffer(self._fid_Y.read(4*BATCH_SIZE),dtype=np.float32)
        self._Y_sup = np.frombuffer(self._fid_Y_sup.read(4*BATCH_SIZE*self._len_tuple),dtype=np.int32)
        if self._X.size<BATCH_SIZE*self._len_tuple*IM_SIZE*IM_SIZE:
            self._fid_X.seek(0)
            self._fid_Y.seek(0)
            self._fid_Y_sup.seek(0)
            self._X = np.frombuffer(self._fid_X.read(4*BATCH_SIZE*self._len_tuple*IM_SIZE*IM_SIZE),dtype=np.float32)
            self._Y = np.frombuffer(self._fid_Y.read(4*BATCH_SIZE),dtype=np.float32)
            self._Y_sup = np.frombuffer(self._fid_Y_sup.read(4*BATCH_SIZE*self._len_tuple),dtype=np.int32)
        
        return (self._X.reshape([-1,IM_SIZE,IM_SIZE]),
                self._Y.reshape([-1]),
                self._Y_sup.reshape([-1]))   
           
