from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


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
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  """
  :return: number of channel c
  """
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    ## scalar or to large to do one-hot
    if i == _MINIMAP_PLAYER_ID or features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    ## categorical
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def preprocess_screen(screen):
  """
  one-hot encode categorical, normalize scalar/player_id inputs
  :return: c * 64 * 64 input layer
  """
  layers = []
  for i in range(len(features.SCREEN_FEATURES)):
    ## scalar or to large to do one-hot
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE or features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    ## categorical
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def screen_channel():
  """
  :return: number of channel c
  """
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    ## scalar or to large to do one-hot
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE or features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    ## categorical
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c


def preprocess_structure(obs):
  layer = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
  layer[obs.observation['available_actions']] = 1
  return layer




