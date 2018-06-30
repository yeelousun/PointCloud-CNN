import tensorflow as tf
import numpy as np
import math
import sys
import os
import random 


import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

from math import sqrt



def placeholder_inputs(batch_size, numkernel, pgnump):
    pgnump_m4 = pgnump * 4
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, numkernel,pgnump, 3))
    pointclouds_pl_m4 = tf.placeholder(tf.float32, shape=(batch_size, numkernel,pgnump_m4, 3))
    pointclouds_kernel = tf.placeholder(tf.float32, shape=(batch_size, numkernel, 3))
    pointclouds_all = tf.placeholder(tf.float32, shape=(batch_size, numkernel, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, pointclouds_pl_m4, pointclouds_kernel, pointclouds_all, labels_pl

def get_model(point_cloud, point_cloud_m4, point_cloud_kernel,point_cloud_all, is_training = True, bn_decay=None):
    """ Classification PointNet, input is BxPGKxPGNx3, output Bx40 """
    if is_training == True:
        dropout_prob_conv = 0.1
        dropout_prob_f = 0.3
    else:
        dropout_prob_conv = 0.0
        dropout_prob_f = 0.0
       
    batch_size = point_cloud.get_shape()[0].value
    PGK = point_cloud.get_shape()[1].value
    PGN = point_cloud.get_shape()[2].value
    PGN_m4 = point_cloud_m4.get_shape()[2].value
    point_cloud_all =  tf.reshape(point_cloud_all, [batch_size, 1, PGK, -1])
       
    ##net_all
    with tf.variable_scope('net_all') as scope:
        net_all_x = tf.slice(point_cloud_all, [0, 0, 0, 0], [-1, -1,-1, 2])
        net_all_y = tf.slice(point_cloud_all, [0, 0, 0, 2], [-1, -1,-1, 1])
        for y_num in range (512): 
            if y_num == 0:
                net_all_y_m = net_all_y
            else :      
                net_all_y_m = tf.concat([net_all_y_m,net_all_y],axis=3)
        neta = conv2d("conv1_1", input=net_all_x, activation=selu, ksize=1, f_in=2, f_out=256)
        neta = conv2d("conv1_2", input=neta, activation=selu, ksize=1, f_in=256, f_out=2048)
        neta = conv2d("conv1_3", input=neta, activation=selu, ksize=1, f_in=2048, f_out=2048)
        neta = conv2d_non("conv1_4", input=neta, activation=selu, ksize=1, f_in=2048, f_out=512)
                  
        loss_a = tf.reduce_mean(tf.square(net_all_y_m - neta),axis=2)
        loss_a = tf.add(loss_a,1)

        out_allfeature = tf.pow(loss_a, -1)
        out_allfeature = tf.multiply(out_allfeature,10.0)
        out_allfeature = tf.maximum(out_allfeature, 0.00001)
        out_allfeature = tf.reshape(out_allfeature, [batch_size, -1])
        out_allfeature = tf.cast(out_allfeature, tf.float32)
    ##net_pc
    with tf.variable_scope('net_pointcloud') as scope:
        net_pc_x = tf.slice(point_cloud, [0, 0, 0, 0], [-1, -1,-1, 2])
        net_pc_y = tf.slice(point_cloud, [0, 0, 0, 2], [-1, -1,-1, 1])
        for y_num in range (512): 
            if y_num == 0:
                net_pc_y_m = net_pc_y
            else :      
                net_pc_y_m = tf.concat([net_pc_y_m, net_pc_y],axis=3)

        netpc = conv2d("conv2_1", input=net_pc_x, activation=selu, ksize=1, f_in=2, f_out=256)
        netpc = conv2d("conv2_2", input=netpc, activation=selu, ksize=1, f_in=256, f_out=2048)
        netpc = conv2d("conv2_3", input=netpc, activation=selu, ksize=1, f_in=2048, f_out=2048)
        netpc = conv2d_non("conv2_4", input=netpc, activation=selu, ksize=1, f_in=2048, f_out=512)
            
        loss_pc = tf.reduce_mean(tf.square(net_pc_y_m - netpc),axis=2)
        loss_pc = tf.add(loss_pc,1)

        out_pcfeature = tf.pow(loss_pc, -1)
        out_pcfeature = tf.multiply(out_pcfeature,10.0)
        out_pcfeature = tf.maximum(out_pcfeature, 0.00001)
        out_pcfeature = tf.reshape(out_pcfeature, [batch_size, 1, PGK, -1])
        out_pcfeature = tf.nn.max_pool(out_pcfeature, ksize=[1, 1, PGK, 1], strides=[1, 1, 1, 1],padding='VALID',name="max_pool_pc")
        #outfeature = tf.reduce_sum(outfeature, 1, keep_dims=False)
        out_pcfeature = tf.reshape(out_pcfeature, [batch_size, -1])
        out_pcfeature = tf.cast(out_pcfeature, tf.float32)        

    ##net_pc_m4
    with tf.variable_scope('net_pointcloud_m4') as scope:
        net_pcm4_x = tf.slice(point_cloud_m4, [0, 0, 0, 0], [-1, -1,-1, 2])
        net_pcm4_y = tf.slice(point_cloud_m4, [0, 0, 0, 2], [-1, -1,-1, 1])
        for y_num in range (512): 
            if y_num == 0:
                net_pcm4_y_m = net_pcm4_y
            else :      
                net_pcm4_y_m = tf.concat([net_pcm4_y_m, net_pcm4_y],axis=3)

        netpcm4 = conv2d("conv3_1", input=net_pcm4_x, activation=selu, ksize=1, f_in=2, f_out=256)
        netpcm4 = conv2d("conv3_2", input=netpcm4, activation=selu, ksize=1, f_in=256, f_out=2048)
        netpcm4 = conv2d("conv3_3", input=netpcm4, activation=selu, ksize=1, f_in=2048, f_out=2048)
        netpcm4 = conv2d_non("conv3_4", input=netpcm4, activation=selu, ksize=1, f_in=2048, f_out=512)

        loss_pcm4 = tf.reduce_mean(tf.square(net_pcm4_y_m - netpcm4),axis=2)
        loss_pcm4 = tf.add(loss_pcm4, 1)

        out_pcm4feature = tf.pow(loss_pcm4, -1)
        out_pcm4feature = tf.multiply(out_pcm4feature,10.0)
        out_pcm4feature = tf.maximum(out_pcm4feature, 0.00001)
        out_pcm4feature = tf.reshape(out_pcm4feature, [batch_size, 1, PGK, -1])
        out_pcm4feature = tf.nn.max_pool(out_pcm4feature, ksize=[1, 1, PGK, 1], strides=[1, 1, 1, 1],padding='VALID',name="max_pool_pcm4")
        #outfeature = tf.reduce_sum(outfeature, 1, keep_dims=False)
        out_pcm4feature = tf.reshape(out_pcm4feature, [batch_size, -1])
        out_pcm4feature = tf.cast(out_pcm4feature, tf.float32)  

    net = tf.concat([out_allfeature,out_pcfeature,out_pcm4feature],axis=1)
    dim = net.get_shape()[1].value

    net = fc('fully_connected1', input=net, activation=selu, n_in=dim, n_out=512, stddev=0.04, bias_init=0.1,
                 weight_decay=0.004)
    net = dropout_selu(net, dropout_prob_f,training=is_training)
    net = fc('fully_connected2', input=net, activation=selu, n_in=512, n_out=256, stddev=0.04, bias_init=0.1,
                 weight_decay=0.004)
    net = dropout_selu(net, dropout_prob_f,training=is_training)

    weights = _variable_with_weight_decay('weights', [256, 40], stddev=1 / 128.0, activation="selu", wd=0.0)
    biases = tf.get_variable(name='biases', shape=[40], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    net = tf.add(tf.matmul(net, weights), biases, name='output')
       
    return net

def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss
'''
def point_group(scope_name,idx_data,data,PGK,PGN):
    with tf.variable_scope(scope_name) as scope:
        batch_size = data.get_shape()[0].value
        data_feature_num =data.get_shape()[-1].value
        pg_data=[]
        pg_data_idx=[]
        for point_idx in range (batch_size):
            rerowcol= idx_data[point_idx]
            real_data = data[point_idx]
            KernelList=random.sample(range(rerowcol.shape[0]),PGK)
            #rerowcol_downsample is downsample point
            rerowcol_downsample=tf.gather(rerowcol,KernelList)
            downdata = tf.expand_dims(rerowcol_downsample,1)
            dis = tf.reduce_sum(tf.abs(tf.subtract(rerowcol,downdata)),reduction_indices=2) 
            _, pg_idx = tf.nn.top_k(tf.negative(dis),k = PGN )
            pgr = tf.gather(real_data,pg_idx)
            pgr = tf.expand_dims(pgr,0)
            rerowcol_downsample = tf.expand_dims(rerowcol_downsample,0)
            if point_idx == 0:
                pg_data = pgr
                pg_data_idx = rerowcol_downsample
            else :
                pg_data = tf.concat([pg_data,pgr],axis=0)
                pg_data_idx = tf.concat([pg_data_idx,rerowcol_downsample],axis=0)

    return pg_data, pg_data_idx
'''
def selu(x, name="selu"):
    """ When using SELUs you have to keep the following in mind:
    # (1) scale inputs to zero mean and unit variance
    # (2) use SELUs
    # (3) initialize weights with stddev sqrt(1/n)
    # (4) use SELU dropout
    """
    with ops.name_scope(name) as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def _variable_with_weight_decay(name, shape, activation, stddev, wd=None):    
    # Determine number of input features from shape
    f_in = np.prod(shape[:-1]) if len(shape) == 4 else shape[0]
    
    # Calculate sdev for initialization according to activation function
    if activation == selu:
        sdev = sqrt(1 / f_in)
    elif activation == tf.nn.relu:
        sdev = sqrt(2 / f_in)
    elif activation == tf.nn.elu:
        sdev = sqrt(1.5505188080679277 / f_in)
    else:
        sdev = stddev
    
    var = tf.get_variable(name=name, shape=shape,
                          initializer=tf.truncated_normal_initializer(stddev=sdev, dtype=tf.float32))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(scope_name, input, activation, ksize, f_in, f_out, bias_init=0.0, stddev=5e-2):
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[ksize, ksize, f_in, f_out], activation=activation,
                                             stddev=stddev)
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [f_out], initializer=tf.constant_initializer(bias_init), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        return activation(pre_activation, name=scope.name)

def conv2d_non(scope_name, input, activation, ksize, f_in, f_out, bias_init=0.0, stddev=5e-2):
    with tf.variable_scope(scope_name) as scope:
        kernel = _variable_with_weight_decay('weights', shape=[ksize, ksize, f_in, f_out], activation=activation,
                                             stddev=stddev)
        conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [f_out], initializer=tf.constant_initializer(bias_init), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)
        return pre_activation


def fc(scope_name, input, activation, n_in, n_out, stddev=0.04, bias_init=0.0, weight_decay=None):
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay('weights', shape=[n_in, n_out], activation=activation, stddev=stddev,
                                              wd=weight_decay)
        biases = tf.get_variable(name='biases', shape=[n_out], initializer=tf.constant_initializer(bias_init),
                                 dtype=tf.float32)
        return activation(tf.matmul(input, weights) + biases, name=scope.name)

def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, 
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))

