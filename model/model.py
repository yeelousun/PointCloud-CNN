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
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, numkernel,pgnump, 3))
    pointclouds_kernel = tf.placeholder(tf.float32, shape=(batch_size, numkernel, 3))
    pointclouds_all = tf.placeholder(tf.float32, shape=(batch_size, numkernel, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, pointclouds_kernel, pointclouds_all, labels_pl

def get_model(point_cloud, point_cloud_kernel,point_cloud_all, is_training = True, bn_decay=None):
    """ Classification PointNet, input is BxPGKxPGNx3, output Bx40 """
    # if is_training == True:
    #     dropout_prob_conv = 0.1
    #     dropout_prob_f = 0.2
    # else:
    #     dropout_prob_conv = 0.0
    #     dropout_prob_f = 0.0
       

    batch_size = point_cloud.get_shape()[0].value
    PGK = point_cloud.get_shape()[1].value
    PGN = point_cloud.get_shape()[2].value
    ##net1
    net1_x = tf.slice(point_cloud, [0, 0, 0, 0], [-1, -1,-1, 2])
    net1_y = tf.slice(point_cloud, [0, 0, 0, 2], [-1, -1,-1, 1])
    for y_num in range (64): 
        if y_num == 0:
            net1_y_num = net1_y
        else :      
            net1_y_num = tf.concat([net1_y_num,net1_y],axis=3)

    net1 = conv2d("conv1_1", input=net1_x, activation=selu, ksize=1, f_in=2, f_out=128)
    net1 = conv2d("conv1_2", input=net1, activation=selu, ksize=1, f_in=128, f_out=64)
    #norm
    net1_norm = tf.sqrt(tf.reduce_sum(tf.square(net1), axis=2))  
    net1_y_norm = tf.sqrt(tf.reduce_sum(tf.square(net1_y_num), axis=2))  
    #cos
    net1cos = tf.reduce_sum(tf.multiply(net1, net1_y_num), axis=2)  
    net1cos_norm = net1cos / (net1_norm * net1_y_norm)  
    net1cos_norm = selu(net1cos_norm)
    #feature1 
    Feature1 = tf.reduce_mean(net1cos_norm, 1, keep_dims=False)
    Feature1 = tf.reshape(Feature1, [batch_size, -1])
    
    net1_f = tf.reshape(net1cos_norm, [batch_size, PGK, -1])
    net2_2, net_idx2 = point_group("group1", point_cloud_kernel,net1_f,32,PGN)
    
    ##net2
    net2_x = tf.slice(net2_2, [0, 0, 0, 0], [-1, -1,-1, 63])
    net2_y = tf.slice(net2_2, [0, 0, 0, 63], [-1, -1,-1, 1])
    for y_num in range (128): 
        if y_num == 0:
            net2_y_num = net2_y
        else :      
            net2_y_num = tf.concat([net2_y_num,net2_y],axis=3)

    net2 = conv2d("conv2_1", input=net2_x, activation=selu, ksize=1, f_in=63, f_out=512)
    net2 = conv2d("conv2_2", input=net2, activation=selu, ksize=1, f_in=512, f_out=256)
    #norm
    net2_norm = tf.sqrt(tf.reduce_sum(tf.square(net2), axis=2))  
    net2_y_norm = tf.sqrt(tf.reduce_sum(tf.square(net2_y_num), axis=2))  
    #cos
    net2cos = tf.reduce_sum(tf.multiply(net2, net2_y_num), axis=2)  
    net2cos_norm = net2cos / (net2_norm * net2_y_norm)  
    net2cos_norm = selu(net2cos_norm)
    #feature2 
    Feature2 = tf.reduce_mean(net2cos_norm, 1, keep_dims=False)
    Feature2 = tf.reshape(Feature2, [batch_size, -1])   
    
    net2_f = tf.reshape(net2cos_norm, [batch_size, 1, 32, -1])
    
    ##net3
    net3_x = tf.slice(net2_f, [0, 0, 0, 0], [-1, -1,-1, 255])
    net3_y = tf.slice(net2_f, [0, 0, 0, 255], [-1, -1,-1, 1])
    for y_num in range (128): 
        if y_num == 0:
            net3_y_num = net3_y
        else :      
            net3_y_num = tf.concat([net3_y_num,net3_y],axis=3)

    net3 = conv2d("conv3_1", input=net3_x, activation=selu, ksize=1, f_in=255, f_out=512)
    net3 = conv2d("conv3_2", input=net3, activation=selu, ksize=1, f_in=512, f_out=256)
    #norm
    net3_norm = tf.sqrt(tf.reduce_sum(tf.square(net3), axis=2))  
    net3_y_norm = tf.sqrt(tf.reduce_sum(tf.square(net3_y_num), axis=2))  
    #cos
    net3cos = tf.reduce_sum(tf.multiply(net3, net3_y_num), axis=2)  
    net3cos_norm = net3cos / (net3_norm * net3_y_norm)
    net3cos_norm = selu(net3cos_norm)
    #feature3
    Feature3 = tf.reduce_mean(net3cos_norm, 1, keep_dims=False)
    Feature3 = tf.reshape(Feature3, [batch_size, -1])

    net = tf.concat([Feature1,Feature2,Feature3],axis=1)
    dim = net.get_shape()[1].value

    net = fc('fully_connected1', input=net, activation=selu, n_in=dim, n_out=256, stddev=0.04, bias_init=0.1,
                 weight_decay=0.004)
    #net = dropout_selu(net, dropout_prob_f,training=is_training)
    net = fc('fully_connected2', input=net, activation=selu, n_in=256, n_out=128, stddev=0.04, bias_init=0.1,
                 weight_decay=0.004)
    #net = dropout_selu(net, dropout_prob_f,training=is_training)

    weights = _variable_with_weight_decay('weights', [128, 40], stddev=1 / 128.0, activation="selu", wd=0.0)
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


def fc(scope_name, input, activation, n_in, n_out, stddev=0.04, bias_init=0.0, weight_decay=None):
    with tf.variable_scope(scope_name) as scope:
        weights = _variable_with_weight_decay('weights', shape=[n_in, n_out], activation=activation, stddev=stddev,
                                              wd=weight_decay)
        biases = tf.get_variable(name='biases', shape=[n_out], initializer=tf.constant_initializer(bias_init),
                                 dtype=tf.float32)
        return activation(tf.matmul(input, weights) + biases, name=scope.name)
'''
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
'''