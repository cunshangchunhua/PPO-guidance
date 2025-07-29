import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
from torch.utils.data.sampler import *
import numpy as np


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        #self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.mean_layer = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        mean = torch.tanh(self.mean_layer(x))
        return mean

    def get_dist(self, state):
        mean=self.forward(state)
        #print(mean)
        log_std = self.log_std.expand_as(mean)
        #print([mean, log_std])
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        #self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        #self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, args.action_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size, episode_limit, N, actor_input_dim), prob.shape(mini_batch_size, episode_limit, N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))
        prob = torch.softmax(self.fc3(x), dim=-1)
        return prob


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        #self.use_value_clip = args.use_value_clip
        self.max_action = 1


        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    # def choose_action(self, s):
    #     s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
    #     with torch.no_grad():
    #         dist = self.actor.get_dist(s)
    #         a = dist.sample()  # Sample the action according to the probability distribution
    #         a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
    #         a_logprob = dist.log_prob(a)  # The log probability density of the action
    #     return a.numpy().flatten(), a_logprob.numpy().flatten()
    def choose_action(self, s, evaluate):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            if evaluate:
                a = torch.clamp(self.actor(s), -self.max_action, self.max_action)
                return a.numpy().flatten(), None
            else:
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a = torch.clamp(a, -self.max_action, self.max_action)  # [-max,max]
                a_logprob = dist.log_prob(a)  # The log probability density of the action
                return a.numpy().flatten(), a_logprob.numpy().flatten()

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            lab = 1 - batch['dw']
            deltas = lab*(batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1])  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae*lab[:, t]
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        """
            Get actor_inputs and critic_inputs
            actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
            critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
        """
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):

                values_now = self.critic(critic_inputs[index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1))
                dist_now = self.actor.get_dist(actor_inputs[index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1))
                #dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                a_logprob_n_now = dist_now.log_prob(batch['a_n'][index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1))  # a_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now.sum(1, keepdim=True) - batch['a_logprob_n'][index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1).sum(1, keepdim=True))
                #ratios = torch.exp(a_logprob_n_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))  # ratios.shape=(mini_batch_size, episode_limit, N)

                surr1 = ratios * adv[index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1)
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1)
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss2=torch.mean(actor_loss)
                actor_loss1= actor_loss.masked_select(batch['mask'][index].reshape(self.mini_batch_size * self.N*self.episode_limit, -1))
                actor_loss3 =torch.mean(actor_loss1)
                self.optimizer_actor.zero_grad()
                actor_loss3.backward()

                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.optimizer_actor.step()

                critic_loss = F.mse_loss(values_now.masked_select(
                    batch['mask'][index].reshape(self.mini_batch_size * self.N * self.episode_limit, -1)),
                    v_target[index].reshape(
                        self.mini_batch_size * self.N * self.episode_limit,
                        -1).masked_select(
                        batch['mask'][index].reshape(
                            self.mini_batch_size * self.N * self.episode_limit, -1)))
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)

        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("./model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
