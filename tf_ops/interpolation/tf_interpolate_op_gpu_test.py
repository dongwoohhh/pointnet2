import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
from tf_interpolate import three_nn, three_interpolate


class InterpolateTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with tf.device('/gpu:0'):
      points = tf.constant(np.random.random((1,8,16)).astype('float32'))  # B x N x C
      #print (points)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      dist, idx = three_nn(xyz1, xyz2)
      weight = tf.ones_like(dist)/3.0
      print(idx)
      points_t = tf.transpose(points, (0, 2, 1))  # B x C x N, 1 x 16 x 8
      print(points_t)
      interpolated_points_t = three_interpolate(points_t, idx, weight)  # 1 x 16 x 128
      print (interpolated_points_t)
      interpolated_points = tf.transpose(interpolated_points_t, (0, 2, 1))
      #print (interpolated_points)
    
      with self.test_session() as sess:
        err = tf.test.compute_gradient_error(points_t, (1,16,8), interpolated_points_t, (1,16,128))
        #err = sess.run(interpolated_points_t)
        #print(err.shape)
        print (err)
        self.assertLess(err, 1e-4) 

if __name__=='__main__':
  """
  with tf.device('/gpu:0'):
      points = tf.constant(np.random.random((1,8,16)).astype('float32'))  # B x N x C
      print (points)
      xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
      xyz2 = tf.constant(np.random.random((1,8,3)).astype('float32'))
      dist, idx = three_nn(xyz1, xyz2)
      weight = tf.ones_like(dist)/3.0
      print(idx)
      points_t = tf.transpose(points, (0, 2, 1))  # B x C x N, 1 x 16 x 8
      print(points_t)
      interpolated_points = three_interpolate(points_t, idx, weight)  # 1 x 16 x 128
      print (interpolated_points)
      interpolated_points = tf.transpose(interpolated_points, (0, 2, 1))
      print (interpolated_points)
  """
  tf.test.main() 
