from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MINIMAP_VISIBILITY = features.MINIMAP_FEATURES.visibility_map.index
_MINIMAP_CAMERA = features.MINIMAP_FEATURES.camera.index

_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index
_SCREEN_VISIBILITY = features.SCREEN_FEATURES.visibility_map.index
_SCREEN_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index


def preprocess_minimap(minimap):
  """
  one-hot encode categorical, normalize scalar/player_id inputs
  :return: c * 64 * 64 input layer
  """
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_RELATIVE:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
    elif i == _MINIMAP_SELECTED or i == _MINIMAP_CAMERA or i == _MINIMAP_VISIBILITY:
      layers.append(minimap[i:i + 1] / features.MINIMAP_FEATURES[i].scale)
  return np.concatenate(layers, axis=0)


def minimap_channel():
  """
  :return: number of channel c
  """
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_SELECTED or i == _MINIMAP_CAMERA or i == _MINIMAP_VISIBILITY:
      c += 1
    elif i == _MINIMAP_PLAYER_RELATIVE:
      c += features.MINIMAP_FEATURES[i].scale
  return c


def preprocess_screen(screen):
  """
  one-hot encode categorical, normalize scalar/player_id inputs
  :return: c * 64 * 64 input layer
  """
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
    elif i == _SCREEN_SELECTED or i == _SCREEN_HIT_POINTS or i == _SCREEN_VISIBILITY:
      layers.append(screen[i:i + 1] / features.SCREEN_FEATURES[i].scale)
    elif i == _SCREEN_PLAYER_RELATIVE:
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
    if i == _SCREEN_SELECTED or i == _SCREEN_UNIT_TYPE or i == _SCREEN_HIT_POINTS or i == _SCREEN_VISIBILITY:
      c += 1
    elif i == _SCREEN_PLAYER_RELATIVE:
      c += features.SCREEN_FEATURES[i].scale
  return c


def preprocess_structure(obs):
  layer = np.zeros([len(actions.FUNCTIONS)], dtype=np.float32)
  layer[obs.observation['available_actions']] = 1
  return layer




