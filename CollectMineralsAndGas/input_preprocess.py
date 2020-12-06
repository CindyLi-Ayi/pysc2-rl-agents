from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_MINIMAP_SELECTED = features.MINIMAP_FEATURES.selected.index
_MINIMAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_SCREEN_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SCREEN_SELECTED = features.SCREEN_FEATURES.selected.index


def preprocess_minimap(minimap):
  return None


def minimap_channel():
  return 0


def preprocess_screen(screen):
  """
  one-hot encode categorical, normalize scalar/player_id inputs
  :return: c * 64 * 64 input layer
  """
  layers = []
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_RELATIVE:
      layer = np.zeros([1, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in [3]:
        indy, indx = (screen[i] == j).nonzero()
        layer[0, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)


def screen_channel():
  return 1


def preprocess_structure(obs):
  return None




