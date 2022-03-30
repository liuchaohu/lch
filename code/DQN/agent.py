#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class LinearDecayScheduler():
    """Set hyper parameters by a step-based scheduler with linear decay values.
    """

    def __init__(self, start_value, max_steps):
        """Linear decay scheduler of hyper parameter.
        Decay value linearly untill 0.
        
        Args:
            start_value (float): start value
            max_steps (int): maximum steps
        """
        assert max_steps > 0
        self.cur_step = 0
        self.max_steps = max_steps
        self.start_value = start_value

    def step(self, step_num=1):
        """Step step_num and fetch value according to following rule:
        return_value = start_value * (1.0 - (cur_steps / max_steps))
        Args:
            step_num (int): number of steps (default: 1)
        Returns:
            value (float): current value
        """
        assert isinstance(step_num, int) and step_num >= 1
        self.cur_step = min(self.cur_step + step_num, self.max_steps)

        value = self.start_value * (1.0 - (
            (self.cur_step * 1.0) / self.max_steps))

        return value


from email import policy
import numpy as np
import torch
from model import AtariModel
import copy

class AtariAgent():
    """Agent of Atari env.

    Args:
        act_dim (int): action space dimension
        total_step (int): total epsilon decay steps
        start_lr (float): initial learning rate
        update_target_step (int): target network update frequency
    """

    def __init__(self, act_dim, start_lr, total_step,
                 update_target_step, gamma):
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.act_dim = act_dim
        self.curr_ep = 1
        self.ep_end = 0.1
        self.lr_end = 0.00001
        self.device = torch.device('cuda' if torch.cuda.
                                   is_available() else 'cpu')

        self.gamma = gamma

        self.policy_net = AtariModel(self.act_dim)
        # self.target_net = AtariAgent(self.act_dim)
        self.target_net = copy.deepcopy(self.policy_net)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        
        # for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()): # 复制参数到目标网路targe_net
        #     target_param.data.copy_(param.data)
        

        self.ep_scheduler = LinearDecayScheduler(1, total_step)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters())

    def sample(self, obs):
        """Sample an action when given an observation, base on the current epsilon value, 
        either a greedy action or a random action will be returned.

        Args:
            obs (np.float32): shape of (3, 84, 84) or (1, 3, 84, 84), current observation

        Returns:
            act (int): action
        """
        explore = np.random.choice([True, False],
                                   p=[self.curr_ep, 1 - self.curr_ep])
        if explore:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)

        self.curr_ep = max(self.ep_scheduler.step(1), self.ep_end)
        return act

    def predict(self, obs):
        """Predict an action when given an observation, a greedy action will be returned.

        Args:
            obs (np.float32): shape of (3, 84, 84) or (1, 3, 84, 84), current observation

        Returns:
            act(int): action
        """
        if obs.ndim == 3:  # if obs is 3 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        # pred_q = self.alg.predict(obs).cpu().detach().numpy().squeeze()
        pred_q = self.policy_net(obs).cpu().detach().numpy().squeeze()

        best_actions = np.where(pred_q == pred_q.max())[0]
        act = np.random.choice(best_actions)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        """Update model with an episode data

        Args:
            obs (np.float32): shape of (batch_size, obs_dim)
            act (np.int32): shape of (batch_size)
            reward (np.float32): shape of (batch_size)
            next_obs (np.float32): shape of (batch_size, obs_dim)
            terminal (np.float32): shape of (batch_size)

        Returns:
            loss (float)
        """
        if self.global_update_step % self.update_target_step == 0:
            self.sync_target()

        self.global_update_step += 1

        reward = np.clip(reward, -1, 1)
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        next_obs = torch.tensor(
            next_obs, dtype=torch.float, device=self.device)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float, device=self.device)
        terminal = torch.tensor(
            terminal, dtype=torch.float, device=self.device)

        loss = self._learn(obs, act, reward, next_obs, terminal)

        # learning rate decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(self.lr_scheduler.step(1), self.lr_end)

        return loss
    
    def _learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 获取Q预测值
        pred_values = self.policy_net(obs)
        action_dim = pred_values.shape[-1]
        action = torch.squeeze(action, axis=-1)
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = torch.nn.functional.one_hot(
            action, num_classes=action_dim)
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        pred_value = pred_values * action_onehot
        #  ==> pred_value = [[3.9]]
        pred_value = torch.sum(pred_value, axis=1, keepdim=True)

        # 从target_model中获取 max Q' 的值，用于计算target_Q
        with torch.no_grad():
            max_v = self.target_net(next_obs).max(1, keepdim=True)
            # print(max_v)
            target = reward + (1.0 - terminal) * self.gamma * max_v.values
        loss = self.mse_loss(pred_value, target)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def sync_target(self):
        # self.policy_net.sync_weights_to(self.target_net)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()): # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)