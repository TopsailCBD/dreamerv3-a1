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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class TeacherActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 encoder_hidden_dims=[256, 128],
                 predictor_hidden_dims = [64, 32],
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0,
                 fixed_std=False,
                 latent_dim = 32,
                 height_dim=187,
                 privileged_dim=3 + 24,
                 history_dim = 42*5,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(TeacherActorCritic, self).__init__()

        activation = get_activation(activation)

        self.latent_dim = latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim

        mlp_input_dim_a = latent_dim + 3 #latent vector + lin vel + command
        # mlp_input_dim_a = num_actor_obs - privileged_dim - height_dim + 3 + latent_dim
        mlp_input_dim_c = num_critic_obs #- privileged_dim - height_dim + height_latent_dim + privileged_latent_dim

        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], latent_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)

        # Linear Vel Predictor
        # predictor_layers = []
        # predictor_layers.append(nn.Linear(latent_dim, predictor_hidden_dims[0]))
        # predictor_layers.append(activation)
        # for l in range(len(predictor_hidden_dims)):
        #     if l == len(predictor_hidden_dims) - 1:
        #         predictor_layers.append(nn.Linear(predictor_hidden_dims[l], 3))
        #     else:
        #         predictor_layers.append(nn.Linear(predictor_hidden_dims[l], predictor_hidden_dims[l + 1]))
        #         predictor_layers.append(activation)
        # self.linear_vel_predictor = nn.Sequential(*predictor_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                # actor_layers.append(nn.Tanh())
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)



        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # self.encoder = Encoder(height_latent_dim,privileged_latent_dim, self.height_dim, self.privileged_dim)

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        # obs_with_command = observations[:, self.privileged_dim:-self.height_dim]
        concat_observations = torch.concat((latent_vector, command),
                                           dim=-1)
        self.update_distribution(concat_observations)
        return self.distribution.sample()

    def get_latent_vector(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        return latent_vector

    def get_linear_vel(self, observations, history, **kwargs):
        latent_vector = self.history_encoder(history)
        linear_vel = latent_vector[:,-3:]
        return linear_vel

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history):
        latent_vector = self.history_encoder(history)
        command = observations[:, self.privileged_dim + 6:self.privileged_dim + 9]
        # obs_with_command = observations[:, self.privileged_dim:-self.height_dim]
        concat_observations = torch.concat((latent_vector, command),
                                           dim=-1)
        actions_mean = self.actor(concat_observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # latent_vector = self.encoder(critic_observations[:, :self.privileged_dim],
        #                              critic_observations[:, -self.height_dim:])
        # concat_observations = torch.concat(
        #     (critic_observations[:, self.privileged_dim:-self.height_dim], latent_vector), dim=-1)
        # value = self.critic(concat_observations)
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class Encoder(nn.Module):
    def __init__(self,
                 height_latent_dim,
                 privileged_latent_dim,
                 height_dim,
                 privileged_dim,
                 ):
        super().__init__()
        self.linear1 = nn.Linear(height_dim, 256)
        self.relu1 = nn.ELU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ELU()
        self.linear3 = nn.Linear(128, height_latent_dim)

        self.linear1_2 = nn.Linear(privileged_dim, 64)
        self.relu1_2 = nn.ELU()
        self.linear2_2 = nn.Linear(64, 32)
        self.relu2_2 = nn.ELU()
        self.linear3_2 = nn.Linear(32, privileged_latent_dim)

    def forward(self, privileged_obs, height):
        x = self.linear1(height)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)

        x_2 = self.linear1_2(privileged_obs)
        x_2 = self.relu1_2(x_2)
        x_2 = self.linear2_2(x_2)
        x_2 = self.relu2_2(x_2)
        x_2 = self.linear3_2(x_2)

        return torch.concat((x, x_2), dim=-1)