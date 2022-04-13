import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
from gym.spaces import Box, Discrete
from gym.utils import seeding
import time
import gym


class IntersectionScenario(gym.Env):
    def __init__(self, visualize=True):
        self.seed(0) # just in case we forget seeding
        
        self.init_ego = Car(Point(20, 20), heading = np.pi/2)
        self.init_ego.velocity = Point(1., 0.)
        self.init_adv = Car(Point(105, 90), heading = np.pi, color='blue')
        self.init_adv.velocity = Point(0., 0.)
        self.visualize = visualize
        
        self.collision_point = Point(20, 90)
        self.target = Point(20, 120)
        self.wall = Point(25, 80)
        self.turn_wall = Point(15, 80)
        
        self.noise_adv_pos = 1.0
        self.noise_adv_vel = 1.0
        self.dt = 0.065
        self.T = 40
        
        self.initiate_world()
        self.reset()
        
    def initiate_world(self):
        self.world = World(self.dt, width = 120, height = 120, ppm = 5, visualize=self.visualize)
        self.world.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25)))
        self.world.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))
        self.world.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))
        self.world.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))

        
    def reset(self):
        self.ego = self.init_ego.copy()
        self.ego.min_speed = 0.
        self.ego.max_speed = 20.
        self.adv = self.init_adv.copy()
        self.adv.min_speed = 3.
        self.adv.max_speed = 12.
        self.ego_reaction_time = 0.6
        self.ego_saw_adv = False
        self.add_noise()

        self.world.reset()
        self.obs_prev_1 = self._get_obs()
        self.obs_prev_2 = self._get_obs()
        
        self.world.add(self.ego)
        self.world.add(self.adv)
        
        return self._get_obs()
        
        def close(self):
            self.world.close()
        
    def add_noise(self):
        self.ego.center += Point(0, 20*self.np_random.rand() - 10)
        self.adv.center += Point(20*self.np_random.rand() - 10, 0)
        self.ego_reaction_time += self.np_random.rand() - 0.5

    @property
    def observation_space(self):
        low = np.array([0, 0, -600 - self.noise_adv_pos/2., self.adv.min_speed - self.noise_adv_vel/2.])
        high= np.array([self.target.y + self.ego.max_speed*self.dt, self.ego.max_speed, 80, self.adv.max_speed + self.noise_adv_vel/2.])
        return Box(low=low, high=high)

    @property
    def action_space(self):
        return Box(low=np.array([-3.5]), high=np.array([2.]))

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def ego_can_see_adv(self):
        if self.ego_saw_adv or self.ego.y >= self.wall.y:
            return True
        furthest_point_seen = self.wall.x + self.wall.x*(self.collision_point.y - self.wall.y) / (self.wall.y - (self.ego.y - self.ego_reaction_time * self.ego.yp))
        self.ego_saw_adv = furthest_point_seen > self.adv.x
        return self.ego_saw_adv
        
    def get_adv_control(self):
        if np.abs(self.adv.xp) > self.adv.max_speed:
            return np.array([0, -2.], dtype=np.float32)
        elif np.abs(self.adv.xp) < self.adv.min_speed:
            return np.array([0, 2.], dtype=np.float32)
        else:
            ttc_ego = (self.collision_point.y - self.ego.y) / np.abs(self.ego.yp + 1e-8)
            ttc_adv = (self.adv.x - self.collision_point.x) / np.abs(self.adv.xp + 1e-8)
            if ttc_adv < ttc_ego:
                return np.array([0, -2.], dtype=np.float32)
            else:
                return np.array([0, 2.], dtype=np.float32)
        
    def get_ego_control(self,policy_no=2):
        ttc_ego = (self.collision_point.y - self.ego.y) / np.abs(self.ego.yp + 1e-8)
        ttc_adv = (self.adv.x - self.collision_point.x) / np.abs(self.adv.xp + 1e-8)
        if policy_no==0: # aggressive
            if ttc_ego < 0.05 or ttc_adv < 0:
                return np.array([0, 1.95 + 0.05*self.np_random.rand()], dtype=np.float32)
            elif ttc_ego < ttc_adv - 0.1 or not self.ego_can_see_adv:
                return np.array([0, np.minimum(2.0, np.maximum(1.2, self.ego.inputAcceleration + self.np_random.rand()*0.2 - 0.1))], dtype=np.float32)
            else:
                return np.array([0, -3.25-np.random.rand()*0.25], dtype=np.float32)
            
        elif policy_no==1: # cautious
            ttw_ego = (self.wall.y - self.ego.y)/np.abs(self.ego.yp + 1e-8)
            if ttc_ego < 0.05 or ttc_adv < 0:
                return np.array([0, 1.95 + 0.05*self.np_random.rand()], dtype=np.float32)
            elif ttw_ego > 1.0 and ttw_ego < 4.5:
                return np.array([0, 0], dtype=np.float32)
            elif ttc_ego < ttc_adv - 0.3 or not self.ego_can_see_adv:
                return np.array([0, np.minimum(1.0, np.maximum(0.4, self.ego.inputAcceleration + self.np_random.rand()*0.2 - 0.1))], dtype=np.float32)
            else:
                return np.array([0, -2.75-np.random.rand()*0.25], dtype=np.float32)
        elif policy_no == 2: # try a left turn!
            ttw_ego = (self.turn_wall.x - self.ego.x)
            if ttc_ego < ttc_adv - 0.5 or not self.ego_can_see_adv:
                return np.array([0, np.minimum(1.0, np.maximum(0.4, self.ego.inputAcceleration + self.np_random.rand()*0.2 - 0.1))], dtype=np.float32)
            elif ttc_adv > 0.3:
                return np.array([0, -2.75-np.random.rand()*0.25], dtype=np.float32)
            elif ttw_ego < 3.60:
                return np.array([0.45, 0.55 + 0.05*self.np_random.rand()], dtype=np.float32) 
            elif self.ego.heading < np.pi:
                return np.array([.10, 0.0], dtype=np.float32) #don't accelerate
            else:
                return np.array([0, 0.2], dtype=np.float32)
        elif policy_no == 3: # try a right turn!
            ttw_ego = (self.wall.x - self.ego.x)
            tty_ego = (self.wall.y - self.ego.y)
            if ttc_ego < ttc_adv - 0.6 or not self.ego_can_see_adv:
                # slowly accelerate?
                return np.array([0, np.minimum(0.7, np.maximum(0.4, self.ego.inputAcceleration + self.np_random.rand()*0.2 - 0.1))], dtype=np.float32)
            if tty_ego > 0:
                return np.array([0, -1.65-np.random.rand()*0.15], dtype=np.float32)
            if ttw_ego > -7.60:
                return np.array([-.46, 0.0], dtype=np.float32) #don't accelerate
            elif self.ego.heading > np.pi:
                return np.array([0.415, 0.55 + 0.05*self.np_random.rand()], dtype=np.float32) # accelerate out 
            else:
                return np.array([0, 0.2], dtype=np.float32)

    @property
    def target_reached(self):
        return self.ego.y >= self.target.y

    @property
    def collision_exists(self):
        return self.ego.collidesWith(self.adv)
    
    @property
    def reached_off_map(self):
        return self.ego.x < 0 or self.ego.x > self.world.width or self.ego.y < 0 or self.ego.y > self.world.height
        
    def step(self, action):
        info = {}
        while type(action) == list:
            action = action[0]
        if action is None:
            ego_action = self.get_ego_control()
        elif action in [1,2,3]:
            ego_action = self.get_ego_control(policy_no=action)
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            ego_action = np.array([0, action], dtype=np.float32)
        adv_action = self.get_adv_control()
        
        self.ego.set_control(*ego_action)
        self.adv.set_control(*adv_action)
        info["previous_state_action"] = [self._get_ext_obs(), ego_action]

        self.obs_prev_2 = self.obs_prev_1
        self.obs_prev_1 = self._get_obs()
        self.world.tick()
        return self._get_obs(), 0, self.collision_exists or self.target_reached or self.world.t >= self.T or self.reached_off_map, info
        
    def _get_obs(self):
        return np.array([self.ego.center.x, self.ego.center.y, self.ego.velocity.x, self.ego.center.y,
        self.adv.center.x, self.adv.center.y, self.adv.velocity.x, self.adv.velocity.y])
    
    def _get_ext_obs(self):
        prev_ego_features_1 = self.obs_prev_1[:4]
        prev_ego_features_2 = self.obs_prev_2[:4]
        return np.concatenate((self._get_obs(), prev_ego_features_1, prev_ego_features_2))


    def render(self, mode='rgb'):
        self.world.render()
