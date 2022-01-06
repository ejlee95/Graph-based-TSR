import tensorflow as tf
import tensorflow.contrib.slim as slim
# import tensorflow.contrib as tc

import numpy as np
import time


def conv(inp, kernel_shape, scope_name, stride=[1,1,1,1], dorelu=True,
        weight_init_fn=tf.random_normal_initializer,
        bias_init_fn=tf.constant_initializer, bias_init_val=0.0, pad='SAME',):

    with tf.variable_scope(scope_name):
        std = 1 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
        std = std ** .5
        weights = tf.get_variable('weights', kernel_shape,
                                  initializer=weight_init_fn(stddev=std))
        # Add ReLU
        if dorelu:
            conv = tf.nn.conv2d(inp, weights, strides=stride, padding=pad)
            print(f"{scope_name} conv: {conv.get_shape()}")
            return slim.batch_norm(
                conv,
                activation_fn=tf.nn.relu,
                fused=True,
                renorm=True,
                scope='bn')
        else:
            biases = tf.get_variable('biases', [kernel_shape[-1]],
                                      initializer=bias_init_fn(bias_init_val))
            conv = tf.nn.conv2d(inp, weights, strides=stride, padding=pad) + biases
            print(f"{scope_name} conv: {conv.get_shape()}")
            return conv

def pool(inp, name=None, kernel=[2,2], stride=[2,2]):
    # Initialize max-pooling layer (default 2x2 window, stride 2)
    kernel = [1] + kernel + [1]
    stride = [1] + stride + [1]
    pool = tf.nn.max_pool(inp, kernel, stride, 'SAME', name=name)
#    print(f"{name} pool: {pool.get_shape()}")
    return pool


# depthwise separable convolution
def dsconv(inp, kernel_shape, scope_name, stride=[1,1,1,1], dorelu=True,
        weight_init_fn=tf.random_normal_initializer,
        bias_init_fn=tf.constant_initializer, bias_init_val=0.0, pad='SAME',):

    # input_shape = inp.get_shape().as_list()
    # cost = int(input_shape[1] * input_shape[2] * kernel_shape[2] \
    #     * (kernel_shape[0] / stride[1] * kernel_shape[1] / stride[2] + kernel_shape[3]))
    # print(f"{scope_name} cost:\t{cost}")

    with tf.variable_scope(scope_name):
        depthwise_filter_shape = kernel_shape[:3] + [1]
        pointwise_filter_shape = [1, 1] + kernel_shape[2:]

        depthwise_filter_std = 1.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
        depthwise_filter_std = depthwise_filter_std ** .5
        depthwise_filter = tf.get_variable('depthwise_filter', depthwise_filter_shape,
                                initializer=weight_init_fn(stddev=depthwise_filter_std))
        pointwise_filter_std = 1.0 / (kernel_shape[2] * kernel_shape[3])
        pointwise_filter_std = pointwise_filter_std ** .5
        pointwise_filter = tf.get_variable('pointwise_filter', pointwise_filter_shape,
                                initializer=weight_init_fn(stddev=pointwise_filter_std))

        # print(f"{depthwise_filter_shape}, std: {depthwise_filter_std}")
        # print(f"{pointwise_filter_shape}, std: {pointwise_filter_std}")

        # Add ReLU
        if dorelu:
            dsconv = tf.nn.separable_conv2d(inp, depthwise_filter, pointwise_filter, strides=stride, padding=pad)
#            print(f"{scope_name} dsconv: {dsconv.get_shape()}")
            return slim.batch_norm(
                dsconv,
                activation_fn=tf.nn.relu6,
                fused=True,
                renorm=True,
                scope='bn')
        else:
            bias = tf.get_variable('bias', [kernel_shape[-1]],
                                      initializer=bias_init_fn(bias_init_val))
            dsconv = tf.nn.separable_conv2d(inp, depthwise_filter, pointwise_filter, strides=stride, padding=pad)
#            print(f"{scope_name} dsconv: {dsconv.get_shape()}")
            return tf.nn.bias_add(dsconv, bias)

def hourglass(conv_fn, inp, n, f, hg_id=0, df=128):
    # Upper branch
    nf = f + df
    up1 = conv_fn(inp, [3, 3, f, f], '%d_%d_up1' % (hg_id, n))

    # Lower branch
    pool1 = pool(inp, '%d_%d_pool' % (hg_id, n))
    low1 = conv_fn(pool1, [3, 3, f, nf], '%d_%d_low1' % (hg_id, n))
    # Recursive hourglass
    if n > 1:
        low2 = hourglass(conv_fn, low1, n - 1, nf, hg_id, df)
    else:
        low2 = conv_fn(low1, [3, 3, nf, nf], '%d_%d_low2' % (hg_id, n))
    low3 = conv_fn(low2, [3, 3, nf, f], '%d_%d_low3' % (hg_id, n))

    up_size = tf.shape(up1)[1:3]
    up2 = tf.image.resize_nearest_neighbor(low3, up_size)
    return up1 + up2


