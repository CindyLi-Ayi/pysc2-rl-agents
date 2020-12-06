from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib import features

from network import build_net
import input_preprocess as PP

########################
## v1: train scv if possible
##     choose idle until no idle->choose a point if have select
########################

class A2CAgent(object):
  def __init__(self, training, msize, ssize, max_episode, original_lr, discount, epsilon, random_range, name='A2CAgent'):
    self.name = name
    self.training = training
    self.summary = []
    self.msize = msize
    self.ssize = ssize
    self.action_size = len(actions.FUNCTIONS)
    self.original_lr = original_lr
    self.lr = self.original_lr
    self.max_episode = max_episode
    self.cur_episode = 0
    self.discount = discount
    self.epsilon = epsilon
    self.random_range = random_range
    self.walked = False


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    self.walked = False
    if self.epsilon[1]>0.1:
      self.epsilon[0] *= 0.9
      self.epsilon[1] *= 0.9
    if self.random_range>1:
      self.random_range *= 0.9
    self.cur_episode += 1
    self.lr = self.original_lr * (1 - 0.9 * self.cur_episode / self.max_episode)


  def build_model(self, dev):
    with tf.variable_scope(self.name) and tf.device(dev):
      ## inputs of networks
      self.screen = tf.placeholder(tf.float32, [None, PP.screen_channel(), self.ssize, self.ssize], name='screen')

      ## build networks
      net = build_net(self.screen)
      self.spatial_action, self.value = net

      ## targets & masks
      self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      ## compute log probability
      spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

      ## policy loss & value loss
      action_log_prob = self.valid_spatial_action * spatial_action_log_prob
      advantage = tf.stop_gradient(self.value_target - self.value)
      policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      value_loss = - tf.reduce_mean(self.value * advantage)
      self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      self.summary.append(tf.summary.scalar('value_loss', value_loss))
      loss = policy_loss + value_loss

      ## RMSProp optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      grads = opt.compute_gradients(loss)
      cliped_grad = []
      for grad, var in grads:
        self.summary.append(tf.summary.histogram(var.op.name, var))
        self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
        grad = tf.clip_by_norm(grad, 10.0)
        cliped_grad.append([grad, var])
      self.train_op = opt.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)


  def step(self, obs):
    ## feed to network
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(PP.preprocess_screen(screen), axis=0)
    feed = {self.screen: screen}
    spatial_action = self.sess.run([self.spatial_action], feed_dict=feed)[0]

    ## choose spatial action
    spatial_action = spatial_action.ravel()
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]
    # print(target, end=' ')
    # print(obs.observation['feature_screen'][5, target[1], target[0]] == 3, end=' ')

    ## epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[1]:
      range = int(self.random_range)
      dy = np.random.randint(-range, range)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-range, range)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    if 490 in obs.observation.available_actions:
      # print('train scv')
      return actions.FunctionCall(490, [[0]])
    if obs.observation['player'][3] < obs.observation['player'][4] and obs.observation['player'][1] > 50:
      return actions.FunctionCall(2, [[0], [25, 25]])
    if not self.walked and len(obs.observation['multi_select']) > 0 and 264 in obs.observation.available_actions:
      # print('harvest')
      # print(target)
      self.walked = True
      return actions.FunctionCall(264, [[0], target])
    if 0 in obs.observation.available_actions and obs.observation['player'][-4] == 0:
      # print('noop')
      return actions.FunctionCall(0, [])
    if 6 in obs.observation.available_actions and obs.observation['player'][-4] > 0:
      # print('select')
      self.walked = False
      return actions.FunctionCall(6, [[1]])

    # print('noop')
    return actions.FunctionCall(0, [])


  def update(self, rbs):
    ## backprop after episode

    ## if not episode end, use newwork to estimate value of last state, else 0
    obs = rbs[-1][-1]
    if obs.last():
      R = 0
    else:
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(PP.preprocess_screen(screen), axis=0)
      feed = {self.screen: screen}
      R = self.sess.run(self.value, feed_dict=feed)[0]

    ## prepare input & actions & Q value target
    screens = []

    value_target = np.zeros([len(rbs)], dtype=np.float32)
    value_target[-1] = R

    valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    spatial_action_selected = np.zeros([len(rbs), self.ssize**2], dtype=np.float32)

    rbs.reverse()
    for i, [obs, action, _] in enumerate(rbs):
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(PP.preprocess_screen(screen), axis=0)
      screens.append(screen)

      reward = obs.reward
      act_id = action.function
      act_args = action.arguments

      value_target[i] = reward + self.discount * value_target[i-1]

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen'):
          idx = act_arg[1] * self.ssize + act_arg[0]
          valid_spatial_action[i] = 1
          spatial_action_selected[i, idx] = 1

    screens = np.concatenate(screens, axis=0)

    ## backprop
    feed = {self.screen: screens,
            self.value_target: value_target,
            self.valid_spatial_action: valid_spatial_action,
            self.spatial_action_selected: spatial_action_selected,
            self.learning_rate: self.lr}
    _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
    self.summary_writer.add_summary(summary, self.cur_episode)


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])
