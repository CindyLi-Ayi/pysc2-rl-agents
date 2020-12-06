from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import importlib

from absl import app
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
import tensorflow as tf

COUNTER = 0

FLAGS = flags.FLAGS

## training parameters
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("continuation", False, "Continuously training.")
flags.DEFINE_float("learning_rate", 5e-3, "Learning rate for training.")
flags.DEFINE_float("discount", 0.99, "Discount rate for future rewards.")
flags.DEFINE_integer("max_episode", int(1000), "Total steps for training.")
flags.DEFINE_integer("snapshot_episode", int(100), "Step for snapshot.")
flags.DEFINE_string("snapshot_path", "./snapshot/", "Path for snapshot.")
flags.DEFINE_string("log_path", "./log/", "Path for log.")
flags.DEFINE_string("device", "0", "Device for training.")
flags.DEFINE_float("epsilon_non_spatial", 0.05, "epsilon of choosing random action")
flags.DEFINE_float("epsilon_spatial", 0.2, "epsilon of choosing random xy location")
flags.DEFINE_integer("random_range", 5, "change of xy location")
flags.DEFINE_integer("mean_episodes", 100, "How many episode for mean")

## game parameters
flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

## agent parameters
flags.DEFINE_string("agent", "a2c_agent.A2CAgent", "Which agent to run.")
flags.DEFINE_enum("agent_race", 'random', sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_enum("bot_race", None, sc2_env.Race._member_names_, "Bot's race.")
flags.DEFINE_enum("difficulty", None, sc2_env.Difficulty._member_names_, "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 1200, "Total agent steps.")



FLAGS(sys.argv)
if FLAGS.training:
  DEVICE = '/cpu:'+FLAGS.device
else:
  DEVICE = '/cpu:0'

LOG = FLAGS.log_path
SNAPSHOT = FLAGS.snapshot_path
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def run_loop(agents, env, max_frames=0):
  start_time = time.time()
  try:
    while True:
      num_frames = 0
      timesteps = env.reset()
      for a in agents:
        a.reset()
      while True:
        num_frames += 1
        last_timesteps = timesteps
        actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
        timesteps = env.step(actions)
        is_done = (num_frames >= max_frames) or timesteps[0].last()
        yield [last_timesteps[0], actions[0], timesteps[0]], is_done
        if is_done:
          break
  except KeyboardInterrupt:
    pass
  finally:
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds" % elapsed_time)


def run_thread(agent, player, map_name, visualize):
  with sc2_env.SC2Env(
      map_name=map_name,
      players=[player],
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.screen_resolution,
          feature_minimap=FLAGS.minimap_resolution),
      step_mul=FLAGS.step_mul,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    replay_buffer = []
    mean_score = 0
    count = 0
    for recorder, is_done in run_loop([agent], env, FLAGS.max_agent_steps):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          global COUNTER
          COUNTER += 1
          agent.update(replay_buffer)
          replay_buffer = []
          if COUNTER % FLAGS.snapshot_episode == 1:
            agent.save_model(SNAPSHOT, COUNTER)
          if COUNTER >= FLAGS.max_episode:
            break
      elif is_done:
        count += 1
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
        mean_score = mean_score + score
        if count % FLAGS.mean_episodes == 0:
          print("your mean score in 100 episode is ", mean_score // 100)
          mean_score = 0


def main(arg):
  maps.get(FLAGS.map)

  agent_classes = []

  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)
  agent_classes.append(agent_cls)
  player = sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], agent_name)

  agent = agent_cls(FLAGS.training,
                    FLAGS.minimap_resolution,
                    FLAGS.screen_resolution,
                    FLAGS.max_episode,
                    FLAGS.learning_rate,
                    FLAGS.discount,
                    [FLAGS.epsilon_non_spatial, FLAGS.epsilon_spatial],
                    FLAGS.random_range)
  agent.build_model(DEVICE)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  agent.setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  run_thread(agent, player, FLAGS.map, FLAGS.render)


if __name__ == "__main__":
  app.run(main)
