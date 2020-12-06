# pysc2-rl-agents

唐浩然，马康祺，李心怡



## Table of Contents

[TOC]

## Run

- 如果对已有参数进行test，需要先解压snapshot文件夹下面的参数文件

```shell
> python -m main --map=MoveToBeacon --training=False
```



## Base model

* 所有minigame都基于跟原始论文中全卷积网络相似的结构，但是用epsilon-greedy的方法进行探索。

  

### Input_preprocess

* 对空间输入（minimap和screen中的每一层），我们对定量变量进行归一化，对分类变量进行one-hot encoding。其中有两个层的分类变量过多，因此也采取归一化的方法。

```python
def preprocess_minimap(minimap):
  """
  one-hot encode categorical, normalize scalar/player_id inputs
  :return: c * 64 * 64 input layer
  """
  layers = []
  for i in range(len(features.MINIMAP_FEATURES)):
    ## scalar or to large to do one-hot
    if i == _MINIMAP_PLAYER_ID or features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    ## categorical
    else:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, 
                        minimap.shape[1], 
                        minimap.shape[2]], 
                       dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)
```

* 对于非空间的信息，我们在代码中留了接口，但是后续没有使用。大部分都是用rule调用。



### Network

* 网络结构与论文中的FCN基本一致，但是缩小了空间信息提取时的channel个数（缩小为原先的一半）。
  * minimap 和 screen 中的空间信息分别通过两个卷积层
  * 非空间信息通过一个全连接层，和tanh激活层
  * 提取后的空间信息channel叠在一起， 再经过一个channel=1的卷积层，最后通过softmax层输出空间参数
  * 提取后的空间和非空间信息横向拼接在一起，再经过一个全连接层，和relu激活层，形成feature层
  * feature层通过一个全连接层，再通过一个softmax层输出非空间动作
  * feature层通过另外一个全连接层，输出Q value

```python
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
```



### A2CAgent

* 在`step`函数里定义了agent如何从obs输入计算出动作，下面是step函数的一部分

  * 对于网络的输出`spatial_action`，选取其最大值
  * 对于网络的输出`non_spatial_action`，选取当前允许选取的动作当中的最大值
  * 如果该动作需要非空间参数，假设为0。大部分情况下都是最优选择
    * 例如，queue参数为0时选取的是‘now’
  * 在`epsilon[0]`的概率下选择随机动作，在`epsilon[1]`的概率下选择随机空间参数

  ```python
  ## choose spatial and non-spatial action
  non_spatial_action = non_spatial_action.ravel()
  valid_actions = obs.observation['available_actions']
  act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
  
  spatial_action = spatial_action.ravel()
  target = np.argmax(spatial_action)
  target = [int(target // self.ssize), int(target % self.ssize)]
  
  ## epsilon greedy exploration
  if self.training and np.random.rand() < self.epsilon[0]:
    act_id = np.random.choice(valid_actions)
  if self.training and np.random.rand() < self.epsilon[1]:
    range = int(self.random_range)
    dy = np.random.randint(-range, range)
    target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
    dx = np.random.randint(-range, range)
    target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))
  
  ## return function
  act_args = []
  for arg in actions.FUNCTIONS[act_id].args:
    if arg.name in ('screen', 'minimap', 'screen2'): ## spatial arg
      act_args.append([target[1], target[0]])
    else:
      act_args.append([0])  ## non-spatial arg
  
  return actions.FunctionCall(act_id, act_args)
  ```

* 在`build_model`里规定了计算loss和更新参数的流程

  * 从目标Q value和网络Q value上计算出advantage
  * 分别计算空间动作的概率和非空间动作在能够采取的非空间动作中的softmax概率，乘以advantage计算出policy loss
  * 网络Q value乘以advantage计算出value loss
  * 二者相加用RMSProp更新参数

  ```python
  ## compute log probability
  spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
  spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
  non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
  valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
  valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
  non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
  non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
  
  ## policy loss & value loss
  action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
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
  ```

- 学习率以如下方式decay

  ```python
  self.lr = self.original_lr * (1 - 0.9 * self.cur_episode / self.max_episode)
  ```

  



### Main

* 包括了一些默认参数

  ```python
  flags.DEFINE_float("learning_rate", 5e-3, "Learning rate for training.")
  flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
  flags.DEFINE_integer("max_episode", int(1000), "Total steps for training.")
  flags.DEFINE_integer("snapshot_episode", int(100), "Step for snapshot.")
  flags.DEFINE_float("epsilon_non_spatial", 0.05, "epsilon of choosing random action")
  flags.DEFINE_float("epsilon_spatial", 0.2, "epsilon of choosing random xy location")
  flags.DEFINE_integer("random_range", 5, "change of xy location")
  flags.DEFINE_integer("mean_episodes", 100, "How many episode for mean")
  ```





## Result for each mini-game







### Collect Minerals and Gas

Baseline: 3351

Best score: 3925

Best mean: 3841

- 总共尝试了以下几个版本：

1. Base model

2. 限制输入空间到`screen_feature.relative`，动作空间到给定的5个有用动作（无动作、选中大本营、造scv、选中空闲scv、采矿）

3. 限制输入空间到`screen_feature.relative`，动作空间到给定的3个有用动作（无动作、选中空闲scv、采矿）

4. 限制输入空间到`screen_feature.relative`中的`NEUTRAL`层，动作空间中非空间动作选择给定（一旦可能就选中大本营并造scv，此外一旦有空闲scv就进行采矿），部分空间参数由网络自己选择（选中大本营空间参数给定，采矿空间参数）。rule部分的代码如下

   ```python
   if 490 in obs.observation.available_actions:
     # print('train scv')
     return actions.FunctionCall(490, [[0]])
   if obs.observation['player'][3] < obs.observation['player'][4] and obs.observation['player'][1] > 50:
     return actions.FunctionCall(2, [[0], [25, 25]])
   if not self.walked and len(obs.observation['multi_select']) > 0 and 264 in obs.observation.available_actions:
     # print('harvest')
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
   ```

- 此外，对网络进行了如下改进和调整：针对该任务，原始网络的对动作空间的探索不够，体现在epsilon和random_range过小。这是因为，该动作当中唯一会改变input layer的动作采矿，如果动作目标没有点击在矿上，scv根本不会动。这就意味着，如果一开始网络的输出即使加上探索也没有可能点击在矿上，网络的训练就无法进行，因此我们设计了递减的epsilon和random_range。

  ```python
  flags.DEFINE_float("epsilon_non_spatial", 1, "epsilon of choosing random action") ## =0 for version 4
  flags.DEFINE_float("epsilon_spatial", 1, "epsilon of choosing random xy location")
  flags.DEFINE_integer("random_range", 32, "change of xy location")
  
  ## 在reset函数中
  if self.epsilon[1]>0.1:
    self.epsilon[1] *= 0.999
  if self.random_range>1:
    self.random_range *= 0.999
  ```

- 其中，前三个版本均无法跑出有效结果，原因总结如下：
  - 当使用原网络时，算力不足，网络比原始paper里的网络小，算的轮数等等都不够
  - v2v3被寄予很大期望，但是仍然受限于”如果网络输出的空间参数不在矿上，那么就一直拿不到分“的困境。当epsilon逐渐减小的时候，就可以发现training的score越来越差。所以实质上网络是在靠随机某一次点到矿上来得分，而不是学到了哪一次是对的。
  - v4按道理来说应该能训练出来，但是我们的训练结果并不稳定。目前snapshot里保存了一份成功的参数。









