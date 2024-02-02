# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
from Envs.v7.multi_spinenvs import spinenvs_handler


class Env:
    def __make_env(self, N_samples, stdz, stda, stdbamp):
        '''
        Initialize the environment exactly as we've done before. Keep all parameters FIXED
        '''
        # Create environment
        envs_handler = spinenvs_handler()

        # Each assignment contains [mean, variance] info
        #std_DisZ = 0.01
        envs_handler.env_dist_pars["DisZ"] = [0.0, stdz**2]
        envs_handler.env_dist_pars["Jint"] = [1, 0]
        envs_handler.env_dist_pars["angle_err"] = [1, stda**2]
        envs_handler.env_dist_pars["tau"] = [1, 0]
        envs_handler.env_dist_pars["Bamp"] = [50, stdbamp**2]
        envs_handler.env_dist_pars
        args_fixed = {
                        "spin":1/2, 
                        "n_spins":2, 
                        "N_composite":1, 
                        "maxreward":30.0, 
                        "printstatements":False, 
                        "only_pipulses":False, 
                        "isNV":False,
                        "instantaneous_pulses":False,
                        "allow_SQpulses":True,
                        "allow_DQpulses":True,
                        "bound_action_duration":False, 
                        "check_uniqueacts":False
                    } 
                    
        envs_handler.args_fixed.update(args_fixed)

        # Generate samples, automatically assigned to envs_handler.dict_samples
        envs_handler.gen_samples(N_samples=N_samples)
        mean_dist = 10
        std_dist = 3
        bounds = [1, None] # limit separation (nm) between bounds
        samples_Jint, dist_samples = envs_handler.gen_Jints_from_separations(mean_dist, 
                                                                            std_dist, 
                                                                            bounds=bounds, 
                                                                            N_samples=N_samples)
        envs_handler.dict_samples['Jint'] = samples_Jint
        envs_handler.gen_spinenvs()
        return envs_handler

    def __get_obs(self):
        pass
    
    def __init__(self, n_spins=100, stdz=0.01, stda=0.05, stdbamp=10, use_antigoal=True, ddiff=False, ignore_reset_start=False):
        self.n_spins = n_spins
        self.stdz = stdz
        self.stda = stda
        self.stdbamp = stdbamp

        self.pulse_env = self.__make_env(n_spins, stdz, stda, stdbamp)


    @property
    def state_size(self):
        pass 

    @property
    def goal_size(self):
        pass

    @property
    def action_size(self):
        pass

    @staticmethod
    def to_tensor(x):
        pass

    @staticmethod
    def to_coords(x):
        pass

    @staticmethod
    def dist(goal, outcome):
        pass

    @property
    def maze(self):
        pass

    @property
    def action_range(self):
        pass

    @property
    def state(self):
        pass

    @property
    def goal(self):
        pass

    @property
    def antigoal(self):
        pass

    @property
    def reward(self):
        pass

    @property
    def achieved(self):
        pass

    @property
    def is_done(self):
        pass

    @property
    def is_success(self):
        pass

    @property
    def d_goal_0(self):
        pass

    @property
    def d_antigoal_0(self):
        pass

    @property
    def next_phase_reset(self):
        pass

    @property
    def sibling_reset(self):
        pass

    def reset(self, state=None, goal=None, antigoal=None):
        pass

    def step(self, action):
        pass


class Env:
    def __init__(self, n=None, maze_type=None, use_antigoal=True, ddiff=False, ignore_reset_start=False):
        self.n = n

        self._mazes = mazes_dict
        self.maze_type = maze_type.lower()

        self._ignore_reset_start = bool(ignore_reset_start)

        # Generate a crazy maze specified by its size and generation seed
        if self.maze_type.startswith('crazy'):
            _, size, seed = self.maze_type.split('_')
            size = int(size)
            seed = int(seed)
            self._mazes[self.maze_type] = {'maze': make_crazy_maze(size, seed), 'action_range': 0.95}

        # Generate an "experiment" maze specified by its height, half-width, and size of starting section
        if self.maze_type.startswith('experiment'):
            _, h, half_w, sz0 = self.maze_type.split('_')
            h = int(h)
            half_w = int(half_w)
            sz0 = int(sz0)
            self._mazes[self.maze_type] = {'maze': make_experiment_maze(h, half_w, sz0), 'action_range': 0.25}


        if self.maze_type.startswith('corridor'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_hallway_maze(corridor_length), 'action_range': 0.95}

        if self.maze_type.startswith('umaze'):
            corridor_length = int(self.maze_type.split('_')[1])
            self._mazes[self.maze_type] = {'maze': make_u_maze(corridor_length), 'action_range': 0.95}

        assert self.maze_type in self._mazes

        self.use_antigoal = bool(use_antigoal)
        self.ddiff = bool(ddiff)

        self._state = dict(s0=None, prev_state=None, state=None, goal=None, n=None, done=None, d_goal_0=None, d_antigoal_0=None)

        self.dist_threshold = 0.15

        self.reset()

    @property
    def state_size(self):
        return 2

    @property
    def goal_size(self):
        return 2

    @property
    def action_size(self):
        return 2

    @staticmethod
    def to_tensor(x):
        return torch.FloatTensor(x)

    @staticmethod
    def to_coords(x):
        if isinstance(x, (tuple, list)):
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            x = x.data.numpy()
        return float(x[0]), float(x[1])

    @staticmethod
    def dist(goal, outcome):
        # return torch.sum(torch.abs(goal - outcome))
        return torch.sqrt(torch.sum(torch.pow(goal - outcome, 2)))

    @property
    def maze(self):
        return self._mazes[self.maze_type]['maze']

    @property
    def action_range(self):
        return self._mazes[self.maze_type]['action_range']

    @property
    def state(self):
        return self._state['state'].view(-1).detach()

    @property
    def goal(self):
        return self._state['goal'].view(-1).detach()

    @property
    def antigoal(self):
        return self._state['antigoal'].view(-1).detach()

    @property
    def reward(self):
        r_sparse = -torch.ones(1) + float(self.is_success)
        r_dense = -self.dist(self.goal, self.state)
        if self.use_antigoal:
            r_dense += self.dist(self.antigoal, self.state)
        if not self.ddiff:
            return r_sparse + torch.clamp(r_dense, -np.inf, 0.0)
        else:
            r_dense_prev = -self.dist(self.goal, self._state['prev_state'])
            if self.use_antigoal:
                r_dense_prev += self.dist(self.antigoal, self._state['prev_state'])
            r_dense -= r_dense_prev
            return r_sparse + r_dense

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d <= self.dist_threshold

    @property
    def d_goal_0(self):
        return self._state['d_goal_0'].item()

    @property
    def d_antigoal_0(self):
        return self._state['d_antigoal_0'].item()

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal, 'antigoal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal}

    def reset(self, state=None, goal=None, antigoal=None):
        if state is None or self._ignore_reset_start:
            s_xy = self.to_tensor(self.maze.sample_start())
        else:
            s_xy = self.to_tensor(state)
        if goal is None:
            if 'square' in self.maze_type:
                g_xy = self.to_tensor(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
            else:
                g_xy = self.to_tensor(self.maze.sample_goal())
        else:
            g_xy = self.to_tensor(goal)

        if antigoal is None:
            ag_xy = self.to_tensor(g_xy)
        else:
            ag_xy = self.to_tensor(antigoal)

        self._state = {
            's0': s_xy,
            'prev_state': s_xy * torch.ones_like(s_xy),
            'state': s_xy * torch.ones_like(s_xy),
            'goal': g_xy,
            'antigoal': ag_xy,
            'n': 0,
            'done': False,
            'd_goal_0': self.dist(g_xy, s_xy),
            'd_antigoal_0': self.dist(g_xy, ag_xy),
        }

    def step(self, action):
        try:
            next_state = self.maze.move(
                self.to_coords(self._state['state']),
                self.to_coords(action)
            )
        except:
            print('state', self.to_coords(self._state['state']))
            print('action', self.to_coords(action))
            raise
        self._state['prev_state'] = self.to_tensor(self._state['state'])
        self._state['state'] = self.to_tensor(next_state)
        self._state['n'] += 1
        self._state['done'] = (self._state['n'] >= self.n) or self.is_success