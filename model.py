import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from torchvision import models, transforms

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
THRESHOLD = 0.5


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input.float())
        return output  # [batch_size,hidden_size,num_unit]

class StateCritic(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # define the encoder and decoder models
        self.fc1 = nn.Conv1d(2*hidden_size, hidden_size, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        # use the probability of unit
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu_(self.fc1(hidden))
        output = F.relu_(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output


class Resnet(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size, output_size):
        super(Resnet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, output_size),
            nn.Sigmoid()
        )

        self.model.to(device)
        self.static_encoder = Encoder(static_size,hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(in_features=hidden_size*2,
                             out_features=hidden_size*2)
        self.fc2 = nn.Linear(in_features=hidden_size*2,
                             out_features=hidden_size*2)
        self.fc3 = nn.Linear(in_features=hidden_size*2,
                             out_features=hidden_size*2)

    def forward(self, static, dynamic):
        def layers_increase(input_hidden):
            input_hidden = input_hidden.permute(0, 2, 1)
            input_hidden1 = self.fc1(input_hidden)
            input_hidden2 = self.fc2(input_hidden)
            input_hidden3 = self.fc3(input_hidden)
            output = torch.stack(
                [input_hidden1, input_hidden2, input_hidden3], dim=1)
            return output.permute(0, 1, 3, 2)

        # embedding
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)
        # [batch_size, num_unit, hidden_size*2]
        input_hidden = torch.cat((static_hidden, dynamic_hidden), dim=1)
        input_batch = layers_increase(input_hidden)
        # calculate
        out = self.model(input_batch)
        # if mask is None: mask = torch.ones_like(out)    
        # out = torch.sigmoid(out+mask.log())
        return out


class RL4UC(nn.Module):
    def __init__(self, static_size, dynamic_size, hidden_size,
                 batch_size, mask_fn=None, update_fn=None, reward_fn=None, alpha=0.8, num_unit=33):
        super(RL4UC, self).__init__()
        self.mask_fn = mask_fn
        self.update_fn = update_fn
        self.alpha = alpha

        self.batch_size = batch_size
        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.hidden_size = hidden_size
        self.reward_fn = reward_fn
        self.num_unit = num_unit

        self.Actor = Resnet(static_size, dynamic_size, hidden_size, num_unit)
        # self.state_critic = StateCritic(static_size, dynamic_size, hidden_size)
        # self.actor_optim = optim.Adam(self.Actor.parameters(), lr=0.00001)
        # self.critic_optim = optim.Adam(self.state_critic.parameters(), lr=0.00001)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, idx, batch_idx, mode='train'):
        start=time.time()
        batch_size, num_static, num_unit = static.shape
        # definition of some variables
        cross_section_size = (batch_size, num_unit)
        # indicate which unit can change state
        mask = torch.ones(cross_section_size, device=device)
        # number of state change
        nOchg = torch.zeros(cross_section_size, device=device)
        # if solving success last time
        success = torch.ones((batch_size), device=device).bool()

        # structures for holding the output sequence
        tour_dynamic = []
        episode_unit_state, episode_logp= [], []
        episode_reward, episode_power, episode_success, episode_true_cost = [], [], [], []
        for moment in range(24):
            # * make the embedding into the actor
            tour_dynamic.append(dynamic.clone().unsqueeze(1))
            probs = self.Actor(static, dynamic)
            probs = torch.where(mask==1,probs,torch.zeros_like(probs))
            # probs = torch.sigmoid(probs+mask.log())

            # state sampling
            if mode == 'train':
                # when train, eplore + exploit
                adjustment=torch.randn(1)/2
                adjustment = (1-self.alpha)*torch.clamp(adjustment,-0.5,0.5)
                adjustment=adjustment.to(device)
                probs_adjusted = self.alpha*probs+adjustment
                probs_adjusted = torch.clamp(probs_adjusted,0.01,0.99)
                m = torch.distributions.bernoulli.Bernoulli(
                    probs=probs_adjusted)
                unit_state = m.sample()  # [256,33]
                logp = m.log_prob(unit_state)  # [256,33]
            else:
                # when evaluate or imitate, just exploit
                ones=torch.ones_like(probs)
                zeros=torch.zeros_like(probs)
                unit_state = torch.where(probs >= THRESHOLD, ones, zeros)
                prob = torch.where(probs >= THRESHOLD, probs, 1-probs)
                logp = prob.log()

            if self.reward_fn is not None:
                last_power = dynamic[:, 2, :]
                # only extract load from one unit is enough
                load = dynamic[:, 3, 0]
                # because all the unit share the same load
                reward, power, success, true_cost = self.reward_fn(unit_state, last_power, load, success)

            # when train or evaluate, dynamic is updated according to past states
            if self.update_fn is not None:
                dynamic, nOchg = self.update_fn(
                    dynamic.clone(), idx, unit_state, power, moment, nOchg)

            if self.mask_fn is not None:
                mask = self.mask_fn(dynamic, unit_state, nOchg)

            episode_logp.append(logp.unsqueeze(1))
            episode_unit_state.append(unit_state.unsqueeze(1))
            episode_reward.append(reward.unsqueeze(1))
            episode_true_cost.append(true_cost.unsqueeze(1))
            episode_power.append(power.unsqueeze(1))
            episode_success.append(success.unsqueeze(1))

            if torch.sum(success)==0: break

        episode_logp = torch.cat(episode_logp, dim=1)
        episode_unit_state = torch.cat(episode_unit_state, dim=1)
        episode_reward = torch.cat(episode_reward, dim=1)
        episode_true_cost = torch.cat(episode_true_cost, dim=1)
        episode_power = torch.cat(episode_power, dim=1)
        episode_success = torch.cat(episode_success, dim=1)
        tour_dynamic = torch.cat(tour_dynamic, dim=1)

        return episode_logp, episode_reward, episode_power, episode_success, tour_dynamic, episode_true_cost


if __name__ == '__main__':
    raise Exception('Cannot be called from main')
