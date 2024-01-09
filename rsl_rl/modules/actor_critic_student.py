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
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class StudentActorCritic(ActorCritic):
    is_recurrent = True
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=256,
                        rnn_num_layers=1,
                        init_noise_std=1.0,
                        height_latent_dim=16,
                        privileged_latent_dim=8,
                        height_dim=49,
                        privileged_dim=3 + 24,
                        dropout = 0.2,  #work well on real robot
                        **kwargs):
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        super().__init__(num_actor_obs=num_actor_obs - privileged_dim - height_dim + height_latent_dim + privileged_latent_dim,
                         num_critic_obs=num_actor_obs - privileged_dim - height_dim + height_latent_dim + privileged_latent_dim,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std)

        self.height_latent_dim = height_latent_dim
        self.privileged_latent_dim = privileged_latent_dim
        self.height_dim = height_dim
        self.privileged_dim = privileged_dim

        activation = get_activation(activation)

        self.memory_a = Memory(num_actor_obs - privileged_dim - height_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_actor_obs - privileged_dim - height_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(rnn_hidden_size, 512))
        encoder_layers.append(activation)
        encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(512, 256))
        encoder_layers.append(activation)
        encoder_layers.append(nn.Dropout(dropout))
        encoder_layers.append(nn.Linear(256, height_latent_dim + privileged_latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations[..., self.privileged_dim:-self.height_dim], masks, hidden_states)
        input_a_shape = input_a.shape
        if (len(input_a_shape) == 3):
            input_a = input_a.flatten(0, 1)
        latent_vector = self.encoder(input_a)
        if(len(observations.shape) == 3):
            observations = unpad_trajectories(observations, masks)
            observations = observations.flatten(0, 1)
        concat_observations = torch.concat((observations[:, self.privileged_dim:-self.height_dim], latent_vector),
                                           dim=-1)
        return super().act(concat_observations)

    def get_actions_and_latent_vectors(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations[..., self.privileged_dim:-self.height_dim], masks, hidden_states)
        input_a_shape = input_a.shape
        if (len(input_a_shape) == 3):
            input_a = input_a.flatten(0, 1)
        latent_vector = self.encoder(input_a)
        if(len(observations.shape) == 3):
            observations = unpad_trajectories(observations, masks)
            observations = observations.flatten(0, 1)
        concat_observations = torch.concat((observations[:, self.privileged_dim:-self.height_dim], latent_vector),
                                           dim=-1)
        return super().act_inference(concat_observations), latent_vector

    def act_inference(self, observations,  masks=None, hidden_states=None):
        input_a = self.memory_a(observations[..., self.privileged_dim:-self.height_dim], masks, hidden_states)
        input_a_shape = input_a.shape
        if (len(input_a_shape) == 3):
            input_a = input_a.flatten(0, 1)
        latent_vector = self.encoder(input_a)
        if (len(observations.shape) == 3):
            observations = unpad_trajectories(observations, masks)
            observations = observations.flatten(0, 1)
        concat_observations = torch.concat((observations[:, self.privileged_dim:-self.height_dim], latent_vector),
                                           dim=-1)
        return super().act_inference(concat_observations)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations[:, self.privileged_dim:-self.height_dim], masks, hidden_states)
        input_c_shape = input_c.shape
        if (len(input_c_shape) == 3):
            input_c = input_c.flatten(0, 1)
        latent_vector = self.encoder(input_c)
        if(len(critic_observations.shape) == 3):
            critic_observations = unpad_trajectories(critic_observations, masks)
            critic_observations = critic_observations.flatten(0, 1)
        concat_observations = torch.concat((critic_observations[:, self.privileged_dim:-self.height_dim], latent_vector),
                                           dim=-1)
        return super().evaluate(concat_observations)
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if (self.hidden_states is not None):
            for i in range(len(self.hidden_states)):
                self.hidden_states[i][..., dones, :] = 0.0