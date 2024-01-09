import numpy as np
from gym import spaces
import torch

class LeggedRobotDreamer:
    def __init__(self, env):
        self._env = env
        self._env.num_commands = 3
        
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)
        
    
    # wraps to gym(openai gym) api, using gym.spaces as spaces
    # TODO: determine the limit of spaces
    @property
    def observation_space(self):
        spaces = {
            # only those used in dreamer is included here.
            "proprioception": spaces.Box(-np.inf, np.inf, (self.num_obs-self._env.num_commands,), dtype=np.float32), # used in dreamer
            # "command": spaces.Box(-np.inf, np.inf, (self._env.num_commands,), dtype=np.float32), # used in agent training
            # "privileged": spaces.Box(-np.inf, np.inf, (self.num_obs,), dtype=np.float32), # not used, to be done as more information
        }
        
        obs_space = spaces.Dict(
            {
                **spaces,
                "is_first": spaces.Box(0, 1, (), dtype=bool),
                "is_last": spaces.Box(0, 1, (), dtype=bool),
                "is_terminal": spaces.Box(0, 1, (), dtype=bool),
            }
        )
        
        return obs_space
    
    @property
    def action_space(self):
        space = spaces.Box(-1, 1, (self.num_actions,), dtype=np.float32)
        space.discrete = False
        return space
    
    def _parse_observations(self,obs):
        privileged, proprioception_a, command, proprioception_b, height = torch.split(obs, [
            self._env.privileged_dim, 
            6,
            3,
            self._env.num_obs-self._env.height_dim-(self._env.privileged_dim)-6-3,
            self._env.height_dim
            ], dim=1)
        
        obs = {}
        obs["proprioception"] = torch.cat((proprioception_a,proprioception_b),dim=1)
        obs["command"] = command
        obs["privileged"] = privileged
        obs["height"] = height
        return obs
    
    # return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states
    # obs_without_command = torch.concat((obs[:, self.env.privileged_dim :self.env.privileged_dim+6], obs[:, self.env.privileged_dim+9 :-self.env.height_dim]), dim=1)
    def step(self, action):
        # obs, reward, done, info = self._env.step(action)
        policy_obs, privileged_obs, reward, reset, extra, _, _ = self._env.step(action)
        # if not self._obs_is_dict:
            # obs = {self._obs_key: obs}
        
        obs = self._parse_observations(policy_obs)
        obs["is_first"] = False
        obs["is_last"] = reset
        obs["is_terminal"] = info.get("is_terminal", False)
        
        done = reset
        info = extra
        
        return obs, reward, done, info
    
    # return obs, privileged_obs
    def reset(self):
        policy_obs, privileged_obs = self._env.reset()
        # if not self._obs_is_dict:
            # obs = {self._obs_key: obs}    
        
        obs = self._parse_observations(policy_obs)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        return obs
    