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
flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("max_episodes", 100, "Max episodes.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
FUNCTIONS = actions.FUNCTIONS

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FLAGS(sys.argv)


class DefeatRoaches(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec, ssize):
        super(DefeatRoaches, self).setup(obs_spec, action_spec)
        self.ssize = ssize
        self.wait=0

    def reset(self):
        super(DefeatRoaches, self).reset()
        self.hasmove = False
        self.queue = True
        self.wait=0

    def find_target(self, roaches, marines):
        marines_xy = np.mean(marines, axis=0)
        roaches_xy = np.mean(roaches, axis=0)

        if marines_xy[1] > roaches_xy[1]: #我在下方 打对方最大的y
            return roaches[-1]
        else:
            return roaches[0]

    def mykey(self, x):#排序用
        return x[1]
    def distance(self, pos1, pos2):
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)

        if FUNCTIONS.Attack_screen.id not in obs.observation.available_actions:
            return FUNCTIONS.select_army('select')

        marines = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        marines.sort(key=self.mykey) #y从小到大排
        self.num = len(marines)
        roaches = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_ENEMY]
        roaches.sort(key=self.mykey)

        #调整位置
        marines_xy = np.mean(marines, axis=0)
        roaches_xy = np.mean(roaches, axis=0)

        if self.num < len(marines):#新加兵后
            self.hasmove = False
            self.queue = True

        self.num = len(marines)



        if not self.hasmove and marines_xy[1] > roaches_xy[1] and marines_xy[1] < 45: #在下方就向下调整
            self.hasmove = True

            return FUNCTIONS.Move_screen('now', [marines_xy[0], 63])

        elif not self.hasmove and marines_xy[1] < roaches_xy[1] and marines_xy[1] > 20 :#在上方就向上调整
            self.hasmove = True

            return FUNCTIONS.Move_screen('now', [marines_xy[0], 0])
        elif roaches != []:
            target = self.find_target(roaches, marines)
            if self.queue:
                self.queue = False
                return FUNCTIONS.Attack_screen('queued', target)
            else:
                return FUNCTIONS.Attack_screen('now', target)

        return FUNCTIONS.no_op()


        """
        if not self.hasmove and marines_xy[1] > roaches_xy[1] and abs(marines_xy[1]-roaches_xy[1]) < 10: #在下方就向下调整
            self.hasmove = True
            return FUNCTIONS.Move_screen('now', [marines_xy[0], min(self.ssize, marines_xy[1]+10)])

        elif not self.hasmove and marines_xy[1] < roaches_xy[1] and abs(marines_xy[1]-roaches_xy[1]) < 10:#在上方就向上调整
            self.hasmove = True
            return FUNCTIONS.Move_screen('now', [marines_xy[0], max(0, marines_xy[1]-10)])
        """




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


###################################################################################

def main():
    """Run agent"""

    agent = DefeatRoaches()
    players = []
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], 'CollectMineralShards'))
    run_thread(agent, players, FLAGS.visualize)

if __name__ == "__main__":
    main()
