import tensorflow as tf
import numpy as np
from tf_interpolate import three_nn, three_interpolate


with tf.device('/gpu:0'):
    points = tf.constant(np.random.random((1,8,16)).astype('float32'))  # B x N x C
    #print (points)
    xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
    xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
    dist, idx = three_nn(xyz1, xyz2)
    weight = tf.ones_like(dist)/3.0
    #print(idx)
    points_t = tf.transpose(points, (0, 2, 1))  # B x C x N, 1 x 16 x 8
    print(points_t.shape)
    interpolated_points = three_interpolate(points_t, idx, weight)  # 1 x 16 x 128
    #print (interpolated_points)
    interpolated_points = tf.transpose(interpolated_points, (0, 2, 1))
    print (interpolated_points.shape)