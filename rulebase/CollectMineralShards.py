from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.lib import units
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.env import available_actions_printer

import sys
import time
from absl import flags
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_enum("agent_race", 'random', sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_bool("visualize", False, "Whether to visualize.")
flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
# flags.DEFINE_string("map", "DefeatZerglingsAndBanelings", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("max_episodes", 100, "Max episodes.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")


FUNCTIONS = actions.FUNCTIONS

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FLAGS(sys.argv)

class CollectMineralShards(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec, ssize):
        super(CollectMineralShards, self).setup(obs_spec, action_spec)
        self.ssize = ssize

    def reset(self):
        super(CollectMineralShards, self).reset()
        self.isReset = True
        self.num_marine = 0#当前选中的marine

    def distance(self, pos1, pos2):
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)

        marines = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        if marines[0].x > marines[1].x:
            marines = [marines[1], marines[0]]
        shards = [[unit.x,unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_NEUTRAL]
        shards_x = [unit.x for unit in obs.observation.feature_units if unit.alliance == _PLAYER_NEUTRAL]
        self.midx_shard = np.mean(shards_x)
        if self.isReset:
            self.midx_shard = np.mean(shards_x)
            self.isReset = False

        now_marine = marines[self.num_marine]
        now_marine_xy = [now_marine.x, now_marine.y]

        if not now_marine.is_selected:
            return FUNCTIONS.select_point("select", now_marine_xy)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:

            if shards:
                min_dist = 10000
                target = None
                for shard in shards:
                    # 0 号marine在左侧
                    if self.num_marine ==0 and shard[0] <= self.midx_shard:
                        dist = self.distance(shard, now_marine_xy)
                        if dist<min_dist:
                            target = shard
                            min_dist=dist

                    elif self.num_marine == 1 and shard[0] > self.midx_shard:
                        dist = self.distance(shard, now_marine_xy)
                        if dist < min_dist:
                            target = shard
                            min_dist=dist

                self.num_marine = 1 - self.num_marine
                if target != None:
                    return FUNCTIONS.Move_screen("now", target)

        return FUNCTIONS.no_op()


def run_loop(agent, env, ssize, max_episodes=0):
    """A run loop to have a single agent and an environment interact."""
    start_time = time.time()
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent.setup(observation_spec, action_spec, ssize)

    total_episodes = 0

    try:
        while total_episodes < max_episodes:
            total_episodes += 1
            timesteps = env.reset()
            agent.reset()
            while True:
                last_timesteps = timesteps
                actions = [agent.step(timesteps[0])]
                timesteps = env.step(actions)
                is_done = timesteps[0].last()
                if is_done:
                    break
            yield [last_timesteps[0], actions[0], timesteps[0]], is_done
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds" % elapsed_time)


def run_thread(agent, players, visualize):
    with sc2_env.SC2Env(map_name=FLAGS.map,
                        step_mul=FLAGS.step_mul,
                        visualize=visualize,
                        players=players,
                        agent_interface_format=sc2_env.parse_agent_interface_format(
                            feature_screen=FLAGS.screen_resolution,
                            feature_minimap=FLAGS.minimap_resolution,
                            use_feature_units=True)) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)

        acc_score = 0

        for recorder, is_done in run_loop(agent, env, FLAGS.screen_resolution, max_episodes=FLAGS.max_episodes):
            obs = recorder[-1].observation
            score = obs["score_cumulative"][0]

            print('Your score is ' + str(score) + '!')

            acc_score += score

        print("Average score is", acc_score, "/", FLAGS.max_episodes, '=', acc_score / FLAGS.max_episodes)



def main():
    """Run agent"""

    agent = CollectMineralShards()
    players = []
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], 'CollectMineralShards'))
    run_thread(agent, players, FLAGS.visualize)

if __name__ == "__main__":
    main()
