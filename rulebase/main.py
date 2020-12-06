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

############################################################################

FLAGS = flags.FLAGS

flags.DEFINE_string('f','','kernel')
flags.DEFINE_enum("agent_race", 'random', sc2_env.Race._member_names_, "Agent's race.")
flags.DEFINE_bool("visualize", False, "Whether to visualize.")
#flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use.")
#flags.DEFINE_string("map", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_string("map", "DefeatZerglingsAndBanelings", "Name of a map to use.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")
flags.DEFINE_integer("max_episodes", 20, "Max episodes.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")

################################################################################

FUNCTIONS = actions.FUNCTIONS

_BANELING = units.Zerg.Baneling
_ZERGLING = units.Zerg.Zergling

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FLAGS(sys.argv)

##############################################################################



##############################################################################
        
class CollectMineralShards(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec, ssize):
        super(CollectMineralShards, self).setup(obs_spec, action_spec)
        self.ssize = ssize

    def reset(self):
        super(CollectMineralShards, self).reset()
        self.isReset = True
        self.selected = 0

    def side(self, pos):
        return self.w1 * pos[0] + self.w2 * pos[1] > self.c

    def dist_square(self, pos1, pos2):
        return (pos1[0]-pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    def step(self, obs):
        super(CollectMineralShards, self).step(obs)

        marines = [unit for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]

        if not marines:
            return FUNCTIONS.no_op()

        if self.isReset:
            mid_x = (marines[0].x + marines[1].x) / 2
            mid_y = (marines[0].y + marines[1].y) / 2
            self.w1 = mid_y - self.ssize / 2
            self.w2 = self.ssize / 2 - mid_x
            self.c = self.ssize / 2 * (mid_y - mid_x)
            
        self.isReset = False
        marine_unit = marines[self.selected]
        marine_xy = [marine_unit.x, marine_unit.y]
        if not marine_unit.is_selected:
            return FUNCTIONS.select_point("select", marine_xy)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_NEUTRAL]

            if minerals:
                min_dist = 2e10
                target = None
                for mineral in minerals:
                    dist = self.dist_square(mineral, marine_xy)
                    if self.side(mineral) == self.side(marine_xy) and dist < min_dist:
                        target = mineral
                        min_dist = dist

                self.selected = 1 - self.selected
                if target!=None:
                    return FUNCTIONS.Move_screen("now", target)

        return FUNCTIONS.no_op()

###################################################################################

class DefeatRoaches(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec, ssize):
        super(DefeatRoaches, self).setup(obs_spec, action_spec)
        self.ssize = ssize

    def reset(self):
        super(DefeatRoaches, self).reset()
        self.isReset = True
        
    def adjust(self, pos):
        return [min(max(pos[0],0),self.ssize-1), min(max(pos[1],0),self.ssize-1)]
    
    def find_target(self, roaches, marines):
        marine_mean = np.mean(marines, axis = 0)
        roaches_mean = np.mean(roaches, axis = 0)
        
        target = roaches[0]
        
        if marine_mean[1] > roaches_mean[1]:
            y = -1
            for roach in roaches:
                if roach[1] > y:
                    target = roach
                    y = roach[1]
        else:
            y = 1000
            for roach in roaches:
                if roach[1] < y:
                    target = roach
                    y = roach[1]
                    
        return target
            

    def step(self, obs):
        super(DefeatRoaches, self).step(obs)

        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            roaches = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_ENEMY]
            marines = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]

            if not roaches:
                return FUNCTIONS.no_op()
            
            target = self.find_target(roaches, marines)
            return FUNCTIONS.Attack_screen("now", target)
        
        if FUNCTIONS.select_army.id in obs.observation.available_actions:
            return FUNCTIONS.select_army("select")
        
        return FUNCTIONS.no_op()
    
###################################################################################

class DefeatZerglingsAndBanelings(base_agent.BaseAgent):
    def setup(self, obs_spec, action_spec, ssize):
        super(DefeatZerglingsAndBanelings, self).setup(obs_spec, action_spec)
        self.ssize = ssize

    def reset(self):
        super(DefeatZerglingsAndBanelings, self).reset()
        self.isReset = True
        self.action_queue = []
        self.isUp = True
        self.isRight = True
    
    def findMax(self, pos_arr, index):
        ans = -100000
        target = [pos_arr[0]]
        for pos in pos_arr:
            if pos[index] > ans:
                ans = pos[index]
                target = [pos]
            elif pos[index] == ans:
                target.append(pos)
        return target
    
    def findMin(self, pos_arr, index):
        ans = 100000
        target = [pos_arr[0]]
        for pos in pos_arr:
            if pos[index] < ans:
                ans = pos[index]
                target = [pos]
            elif pos[index] == ans:
                target.append(pos)
        return target
    
    def findMean(self, pos_arr):
        _mean = np.mean(pos_arr, axis = 0)
        return _mean
    
    def adjust(self, pos):
        return [min(max(pos[0],0),self.ssize-1), min(max(pos[1],0),self.ssize-1)]

    def step(self, obs):
        super(DefeatZerglingsAndBanelings, self).step(obs)
        
        marines = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.alliance == _PLAYER_SELF]
        banelings = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == _BANELING]
        zerglings = [[unit.x, unit.y] for unit in obs.observation.feature_units if unit.unit_type == _ZERGLING]
        
        if not marines:
            return FUNCTIONS.no_op()
        
        if self.isReset:
            if(self.findMean(marines)[0] > self.findMean(zerglings)[0]):
                self.isRight = True
            else:
                self.isRight = False
                
            if(self.findMean(marines)[1] < self.findMean(zerglings)[1]):
                self.isUp = True
            else:
                self.isUp = False
            
            if self.isUp:
                select_unit = self.findMin(marines, 1)[0]
                move_target = self.findMean(self.findMin(banelings, 1))
            else:
                select_unit = self.findMax(marines, 1)[0]
                move_target = self.findMean(self.findMax(banelings, 1))
            
            x1, x2 = self.findMin(marines,0)[0][0], self.findMax(marines,0)[0][0]
            y1, y2 = self.findMin(marines,1)[0][1], self.findMax(marines,1)[0][1]
            
            self.action_queue.append(FUNCTIONS.select_point("select", select_unit))
            self.action_queue.append(FUNCTIONS.Move_screen("now", move_target))
            self.action_queue.append(FUNCTIONS.select_rect("select", [min(x1,x2),min(y1,y2)], [max(x1,x2),max(y1,y2)]))
            for i in range(80):
                self.action_queue.append(FUNCTIONS.no_op())
            
        self.isReset = False
        if len(self.action_queue) > 0:
            action = self.action_queue[0]
            self.action_queue = self.action_queue[1:]
            return action

        if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
            if self.isRight:
                if banelings:
                    target = self.findMax(self.findMax(banelings, 0), 1)[0]
                elif zerglings:
                    target = self.findMax(self.findMax(zerglings, 0), 1)[0]
                else:
                    return FUNCTIONS.no_op()
            else:
                if banelings:
                    target = self.findMax(self.findMin(banelings, 0), 1)[0]
                elif zerglings:
                    target = self.findMax(self.findMin(zerglings, 0), 1)[0]
                else:
                    return FUNCTIONS.no_op()
            
            return FUNCTIONS.Attack_screen("now", target)
        
        #if FUNCTIONS.select_army.id in obs.observation.available_actions:
        #    return FUNCTIONS.select_army("add")
        
        return FUNCTIONS.no_op()

###################################################################################

def run_loop(agent, env, ssize, max_episodes = 0):
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

def run_thread(agent, players,map_name, visualize):
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
        
        for recorder, is_done in run_loop(agent, env, FLAGS.screen_resolution, max_episodes = FLAGS.max_episodes):
            obs = recorder[-1].observation
            score = obs["score_cumulative"][0]
            
            print('Your score is '+str(score)+'!')
            
            acc_score += score
        
        print("Average score is", acc_score, "/", FLAGS.max_episodes, '=', acc_score / FLAGS.max_episodes)

###################################################################################

def main():
    """Run agent"""
    
    if FLAGS.map == "CollectMineralShards":
        agent = CollectMineralShards()
    elif FLAGS.map == "DefeatRoaches":
        agent = DefeatRoaches()
    elif FLAGS.map == "DefeatZerglingsAndBanelings":
        agent = DefeatZerglingsAndBanelings()
    players=[]
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race], 'fcn'))

    run_thread(agent, players, FLAGS.map, FLAGS.visualize)
    
main()