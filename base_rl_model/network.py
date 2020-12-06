from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

def build_net(minimap, screen, structure, num_action):
  """
  same network structure as Fully Convolution Network in paper
  :return: spatial_action, non_spatial_action, Q-value
  """
  ## feature extraction
  minimap_conv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                         num_outputs=4,
                         kernel_size=5,
                         stride=1,
                         scope='minimap_conv1')
  minimap_conv2 = layers.conv2d(minimap_conv1,
                         num_outputs=8,
                         kernel_size=3,
                         stride=1,
                         scope='minimap_conv2')
  screen_conv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                         num_outputs=4,
                         kernel_size=5,
                         stride=1,
                         scope='screen_conv1')
  screen_conv2 = layers.conv2d(screen_conv1,
                         num_outputs=8,
                         kernel_size=3,
                         stride=1,
                         scope='screen_conv2')
  structure_fc = layers.fully_connected(layers.flatten(structure),
                                   num_outputs=64,
                                   activation_fn=tf.tanh,
                                   scope='structure_fc')

  ## spatial action branch
  feature_conv = tf.concat([minimap_conv2, screen_conv2], axis=3)
  spatial_action = layers.conv2d(feature_conv,
                                 num_outputs=1,
                                 kernel_size=1,
                                 stride=1,
                                 activation_fn=None,
                                 scope='spatial_action')
  spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

  ## non spatial actions & Q-value branch
  feature_fc = tf.concat([layers.flatten(minimap_conv2), layers.flatten(screen_conv2), structure_fc], axis=1)
  feature_fc = layers.fully_connected(feature_fc,
                                   num_outputs=256,
                                   activation_fn=tf.nn.relu,
                                   scope='feature_fc')
  non_spatial_action = layers.fully_connected(feature_fc,
                                              num_outputs=num_action,
                                              activation_fn=tf.nn.softmax,
                                              scope='non_spatial_action')
  value = tf.reshape(layers.fully_connected(feature_fc,
                                            num_outputs=1,
                                            activation_fn=None,
                                            scope='value'), [-1])

  return spatial_action, non_spatial_action, value
