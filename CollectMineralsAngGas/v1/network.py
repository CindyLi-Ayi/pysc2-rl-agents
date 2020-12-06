from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

def build_net(screen):
  """
  same network structure as Fully Convolution Network in paper
  :return: spatial_action, non_spatial_action, Q-value
  """
  ## feature extraction
  screen_conv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=16,
                         kernel_size=5,
                         stride=1,
                         scope='screen_conv1')
  screen_conv2 = layers.conv2d(screen_conv1,
                         num_outputs=32,
                         kernel_size=3,
                         stride=1,
                         scope='screen_conv2')

  ## spatial action branch
  spatial_action = layers.conv2d(screen_conv2,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  ## Q-value branch
  feature_fc = layers.fully_connected(layers.flatten(screen_conv2),
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feature_fc')
  value = tf.reshape(layers.fully_connected(feature_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, value
