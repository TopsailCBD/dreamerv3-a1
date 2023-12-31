# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import json


name = ['forward', 'forward2', 'backward', 'lateralleft', 'lateralright', 'turnleft', 'turnright', 'mix'][5]
lin_vel_x = {
            'forward': [0.4, 0.4],
            'forward2': [0.8, 0.8],
            'backward': [-0.6,-0.6],
            'lateralleft': [-0.0,-0.0],
            'lateralright': [-0.0,-0.0],
            'turnleft': [0.6, 0.6],
            'turnright': [0.6, 0.6],
            'mix': [-1.0,1.0],
        }[name]

lin_vel_y = {
            'forward': [-0.0,-0.0],
            'forward2': [-0.0,-0.0],
            'backward': [-0.0,-0.0],
            'lateralleft': [0.5,0.5],
            'lateralright': [-0.5,-0.5],
            'turnleft': [0.0, 0.0],
            'turnright': [0.0, 0.0],
            'mix': [-1.0,1.0],
        }[name]

ang_vel_yaw = {
            'forward': [-0.0,-0.0],
            'forward2': [-0.0,-0.0],
            'backward': [-0.0,-0.0],
            'lateralleft': [-0.0,-0.0],
            'lateralright': [-0.0,-0.0],
            'turnleft': [0.5, 0.5],
            'turnright': [-0.5, -0.5],
            'mix': [-1.0,1.0],
        }[name]

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    # env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.heading_command = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.randomize_restitution = False

    env_cfg.domain_rand.friction_range = [1.0, 1.0]
    env_cfg.domain_rand.restitution_range = [0.5, 0.5]
    env_cfg.domain_rand.added_mass_range = [0., 0.]  # kg
    env_cfg.domain_rand.com_pos_range = [-0.0, 0.0]

    env_cfg.domain_rand.randomize_action_latency = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    # env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    # env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_motor_strength = False

    train_cfg.runner.amp_num_preload_transitions = 1

    env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.terrain_proportions = [0.0, 0, 0, 0, 0, 1.0]
    # env_cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0, 0]
    env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0]
    # env_cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0]
    # env_cfg.terrain.terrain_proportions = [0, 0, 0.0, 0, 1.0]
    # env_cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0, 0]

    env_cfg.commands.ranges.lin_vel_x = lin_vel_x
    env_cfg.commands.ranges.lin_vel_y = lin_vel_y
    env_cfg.commands.ranges.ang_vel_yaw = ang_vel_yaw

    env_cfg.commands.resampling_time = 2

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = 'teacher_4_rough_fat0.5_action_latency_0-0.02'
    train_cfg.runner.checkpoint = 20000
    ppo_runner, train_cfg = task_registry.make_teacher_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    record_amp_states = []

    history_length = 5
    trajectory_history = torch.zeros(size=(env.num_envs, history_length, env.num_obs -
                                            env.privileged_dim - env.height_dim - 3), device = env.device)
    obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                        obs[:, env.privileged_dim + 9:-env.height_dim]), dim=1)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)


    for i in range(551):
        history = trajectory_history.flatten(1).to(env.device)
        actions = policy(obs.detach(), history.detach())

        amp_state = env.get_full_amp_observations()
        record_amp_states.append(amp_state[8:9])

        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        obs_without_command = torch.concat((obs[:, env.privileged_dim:env.privileged_dim + 6],
                                            obs[:, env.privileged_dim + 9:-env.height_dim]),
                                           dim=1)
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported',
                                        'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            lootat = env.root_states[8, :3]
            camara_position = lootat.detach().cpu().numpy() + [0, -1, 0]
            env.set_camera(camara_position, lootat)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        # elif i==stop_state_log:
        #     logger.plot_states()
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()
    record_amp_states = torch.concat(record_amp_states, dim=0).detach().cpu().numpy().tolist()
    print(len(record_amp_states))

    data = {
        "LoopMode": "Wrap",
        "FrameDuration": 0.02,
        "MotionWeight": 1.0,

        "Frames":
            record_amp_states[50:]
    }

    with open('datasets/rl_motions/' + name + '.txt', 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = True
    args = get_args()
    args.rl_device = args.sim_device
    play(args)
