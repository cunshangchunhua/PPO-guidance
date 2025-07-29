import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from make_env import make_env
import gym
from missile import acquisition, get_seeker_state, pn_guidance, rK4_step, missile_accel, target_accel, get_reward,get_reward1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani

class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):

        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed

        # Create env
        self.args.N = 1  # The number of agents
        self.args.obs_dim = 2  # The dimensions of an agent's observation space
        self.args.action_dim = 2  # The dimensions of an agent's action space
        self.args.state_dim = 6  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        #self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        p = np.empty((0, 2))
        self.missile_win = 0
        while self.total_steps < 1000:
            self.agent_n.actor.load_state_dict(torch.load("./model_actor/step_2844000.pth"))
            # for i in range(1, 11):
            self.missile_win = self.run_episode_mpe(evaluate=True)
            self.total_steps += 1
        pp = np.array([[self.total_steps, self.missile_win]])
        p = np.vstack((p, pp))
        np.savetxt("win_num10_80%-mean.txt", p, fmt='%f', delimiter=',')

    def run_episode_mpe(self, evaluate=False):
        RT_0x=np.random.uniform(4000, 4200)
        RT_0y = np.random.uniform(4000, 4200)
        #print(RT_0y)
        RT_0 = np.array([RT_0x, RT_0y, 1000])
        #RT_0 = np.array([2000, 1500, 100])
        VT_0 = 200
        psi_t_0 = 0.2
        gma_t_0 = 0
        target_state_0 = np.array([*RT_0, VT_0, psi_t_0, gma_t_0])

        # Target Maneuver, 0 = line, 1 = circle, 2 = Helix, 3 = loop
        TARGET_MANEUVER_NUM = 0

        # missile starts at the same spot with the same velocity
        #RM_0= np.random.uniform(0, 1000, 3)
        RM_0 = np.array([0, 0, 0])

        VM_0 = 400
        engage_params = acquisition(RT_0, RM_0)
        psi_m_0 = engage_params[0]
        gma_m_0 = engage_params[1]

        missile_state_0 = np.array([*RM_0, VM_0, psi_m_0, gma_m_0])
        seeker_state_0 = get_seeker_state(target_state_0, missile_state_0)

        t = 0
        dt = 0.05
        missile_state = missile_state_0

        target_state = target_state_0
        t_sim = [t]
        missile_sim = np.array(missile_state)  # ACTUAL MISSILE STATE
        target_sim = np.array(target_state)  # ACTUAL TARGET STATE
        distance = np.linalg.norm(missile_state[:3] - target_state[:3])

        distace_sim = [distance]

        seeker_state = seeker_state_0

        obs_n =seeker_state

        eps =10.  # impact distance parameter
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        for episode_step in range(self.args.episode_limit):

            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)

            commanded_accel = pn_guidance(seeker_state, a_n)
            missile_state = rK4_step(missile_state, missile_accel, dt, params=commanded_accel)
            target_state = rK4_step(target_state, target_accel, dt, params=TARGET_MANEUVER_NUM)
            seeker_state = get_seeker_state(target_state, missile_state)
            distance = np.linalg.norm(missile_state[:3] - target_state[:3])

            t = t + dt
            t_sim.append(t)
            missile_sim = np.vstack((missile_sim, missile_state))

            target_sim = np.vstack((target_sim, target_state))
            distace_sim.append(distance)

            Vc_track = -(seeker_state[0] * seeker_state[3] + seeker_state[1] * seeker_state[4] + seeker_state[2] *
                         seeker_state[5]) / np.linalg.norm(seeker_state[:3])



            if distance < eps:
                done_n = True
                self.missile_win += 1
                # print("win")
                print([distance])
            else:
                done_n = False
            obs_n = (seeker_state)

            if done_n:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')  # 创建三维坐标轴

                start_object2 = target_sim[0, 0:3]
                start_object1 = missile_sim[0, 0:3]

                ax.plot3D(target_sim[:, 0], target_sim[:, 1], target_sim[:, 2],
                          linewidth=1.5, linestyle='-.', color='maroon',
                          label='target')

                # 绘制导弹1的轨迹
                ax.plot3D(missile_sim[:, 0], missile_sim[:, 1], missile_sim[:, 2],
                          linewidth=1.5, linestyle='-.', color='blue',
                          label='missile1')

                plt.plot(start_object1[0], start_object1[1], start_object1[2],marker='s', color='red', markersize=5,
                         label='Missile Base')
                plt.plot(start_object2[0], start_object2[1], start_object2[2],  marker='o', color='blue', markersize=5, label='Target Base')

                # 设置坐标轴标签
                ax.set_xlabel("X 轴")
                ax.set_ylabel("Y 轴")
                ax.set_zlabel("Z 轴")

                # 设置图例
                ax.legend()

                # 显示图表
                plt.show()
                #plt.pause(1)  # 显示秒数
                plt.close()
                break
        return self.missile_win

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=600, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=12000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.02, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()
    seed = [5]
    for i in range(len(seed)):
        seed_ = seed[i]
        runner = Runner_MAPPO_MPE(args, env_name="missile_nav_testing", number=1, seed=seed_)
        runner.run()
    # runner = Runner_MAPPO_MPE(args, env_name="missile_nav_testing", number=1, seed=5)
    # runner.run()


