import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        #self.action_var = action_std_init * action_std_init
        # encoder
        self.encoder = nn.Sequential(
                            nn.Conv2d(input_channel, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=4, dilation=4, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            )
        self.actor = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,  padding_mode='zeros'),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=64, out_channels=action_dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
                            nn.Softmax(dim=1),
                        )
        # critic
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=3, dilation=3, bias=True, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, bias=True, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, padding_mode='zeros'),
                    )
        self.encoder.apply(self.init_w)
        self.actor.apply(self.init_w)
        self.critic.apply(self.init_w)

    def forward(self, state):
        action_probs = torch.clip(self.actor(self.encoder(state)),1e-13,1-11e-13)
        prob = action_probs.permute(0, 2, 3, 1)
        state_values = self.critic(self.encoder(state))
        #dist = Normal(action_mean, self.action_var)
        return prob, state_values


    def init_w(self, m):
        if not isinstance(m, nn.Conv2d):
            pass
        else:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            m.bias.data.fill_(0.01)

class Buffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data = []

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self): # Convert list of transactions into tensors
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst= [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.extend(s)
            a_lst.extend(a)
            r_lst.extend(r)
            s_prime_lst.extend(s_prime)
            prob_a_lst.extend(prob_a)
            done_mask = 0 if done else 1
            done_lst.extend([done_mask])


        s, a, r, s_prime, prob_a, done_mask = torch.tensor(np.array(s_lst), dtype=torch.float).cuda(), torch.tensor(np.array(a_lst)).cuda(), \
                                          torch.tensor(np.array(r_lst), dtype=torch.float).cuda(), torch.tensor(np.array(s_prime_lst), dtype=torch.float).cuda(), \
                                           torch.tensor(np.array(prob_a_lst)).cuda(), torch.tensor(np.array(done_lst), dtype=torch.float).cuda()


        self.data = []
        a = a.unsqueeze(1)
        prob_a = prob_a.unsqueeze(1)


        return s, a, r, s_prime, prob_a, done_mask.repeat_interleave(self.batch_size)

def train_epoch(model, buffer, optimizer, batch_size, gamma, K_epochs, eps_clip, action_var):
    MseLoss = nn.MSELoss()
    s, a, r, s_prime, prob_a, done_mask = buffer.make_batch()
    rewards = []
    discounted_reward = 0
    for time in range(int(len(r) / batch_size) - 1, -1, -1):
        #if time == int(len(r) / batch_size) - 1:
            #discounted_reward = r[time * batch_size:(time + 1) * batch_size]*0
        discounted_reward = gamma * discounted_reward + r[time * batch_size:(time + 1) * batch_size]
        discounted_reward_np = discounted_reward.detach().cpu().numpy()
        #discounted_reward_np = discounted_reward.numpy()
        for drn in discounted_reward_np[::-1]:
            rewards.insert(0, drn)

    rewards = torch.tensor(np.array(rewards), dtype=torch.float).cuda()

    for _ in range(K_epochs):
        prob_old, state_values = model(s)
        prob_old = prob_old.permute(0,3,1,2)
        logprobs = torch.log(prob_old.gather(1,a))
        ratios = torch.exp(logprobs - prob_a)
        entropy = (- prob_old*torch.log(prob_old)).sum(1).mean()
        advantages = rewards - state_values.detach()
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        # final loss of clipped objective PPO
        loss = -torch.min(surr1, surr2) + 0.5*MseLoss(state_values, rewards) - 0.01*entropy

        # take gradient step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
