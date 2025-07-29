import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_limit+1, self.N]),

                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N, self.action_dim]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N, self.action_dim]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'dw': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'mask': np.empty([self.batch_size, self.episode_limit, self.N], dtype=bool)
                       }
        self.episode_num = 0
        self.max_episode_len = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        #self.buffer['v_next'][self.episode_num][episode_step] = v_next
        self.buffer['a_n'][self.episode_num][episode_step] = a_n
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n
        self.buffer['dw'][self.episode_num][episode_step] = False
        self.buffer['mask'][self.episode_num][episode_step]=True
        #self.episode_num += 1

    def tianjia(self, episode_step, n,r_n,Flag):
        self.buffer['obs_n'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['obs_n'][self.episode_num][episode_step]
        #print(self.buffer['obs_n'][self.episode_num][0:n-1])
        self.buffer['s'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['s'][self.episode_num][episode_step]
        self.buffer['v_n'][self.episode_num][episode_step+1:self.episode_limit+2] = self.buffer['r_n'][self.episode_num][episode_step]
        #self.buffer['v_next'][self.episode_num][episode_step + 1] = self.buffer['s'][self.episode_num][0:n - 1]
        self.buffer['a_n'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['a_n'][self.episode_num][episode_step]
        self.buffer['a_logprob_n'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['a_logprob_n'][self.episode_num][episode_step]
        self.buffer['r_n'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['r_n'][self.episode_num][episode_step+1]
        self.buffer['done_n'][self.episode_num][episode_step+1:self.episode_limit+1] = self.buffer['done_n'][self.episode_num][episode_step]
        self.episode_num += 1

    def tianjiap(self,episode_step, n,r_n,Flag):
        #self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = Flag
        self.buffer['done_n'][self.episode_num][self.episode_limit-1] = True
        self.episode_num += 1

    def tianjiap1(self,episode_step, n,r_n,Flag):
        #self.buffer['r_n'][self.episode_num][episode_step] = r_n
        #self.buffer['done_n'][self.episode_num][episode_step] = Flag
        #self.buffer['dw'][self.episode_num][episode_step + 1:self.episode_limit + 1] = True
        self.episode_num += 1

    def tianjiap2(self,episode_step, n,r_n,Flag):
        #self.buffer['r_n'][self.episode_num][episode_step] = r_n
        #self.buffer['done_n'][self.episode_num][episode_step] = Flag
        es=1e-7
        self.buffer['obs_n'][self.episode_num][episode_step + 1:self.episode_limit + 1] = es
        # print(self.buffer['obs_n'][self.episode_num][0:n-1])
        self.buffer['s'][self.episode_num][episode_step + 1:self.episode_limit + 1] = es
        self.buffer['v_n'][self.episode_num][episode_step + 1:self.episode_limit+2] = es
        # self.buffer['v_next'][self.episode_num][episode_step + 1] = self.buffer['s'][self.episode_num][0:n - 1]
        self.buffer['a_n'][self.episode_num][episode_step + 1:self.episode_limit + 1] = es
        self.buffer['a_logprob_n'][self.episode_num][episode_step + 1:self.episode_limit + 1] = es
        self.buffer['r_n'][self.episode_num][episode_step + 1:self.episode_limit + 1] = es
        self.buffer['done_n'][self.episode_num][episode_step + 1:self.episode_limit + 1] = 0
        self.buffer['dw'][self.episode_num][episode_step + 1:self.episode_limit + 1] = True
        self.buffer['mask'][self.episode_num][episode_step + 1:self.episode_limit + 1] = False
        self.episode_num += 1

    # def store_transition(self,  obs_n, s, s_next,  a_n, a_logprob_n, r_n, done_n):
    #     self.buffer['obs_n'][self.episode_num] = obs_n
    #     self.buffer['s'][self.episode_num] = s
    #     self.buffer['s_next'][self.episode_num] = s_next
    #     self.buffer['a_n'][self.episode_num] = a_n
    #     self.buffer['a_logprob_n'][self.episode_num] = a_logprob_n
    #     self.buffer['r_n'][self.episode_num] = r_n
    #     self.buffer['done_n'][self.episode_num] = done_n
    #     self.episode_num += 1
    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n

        #self.episode_num += 1



    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():

            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float)
            elif key =='mask':
                batch[key] = torch.tensor(self.buffer[key], dtype=bool)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float)
        return batch
