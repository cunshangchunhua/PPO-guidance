
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
#from make_env import make_env
#import gym
from missile import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as ani
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import scipy.io

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
        # Set random seedC:\Users\Shah\Desktop\针对某策略的某研究（公开）\典型场景的博弈对抗仿真分析模型（公开）\PPO\ppo-python-L-固定\venv

        # Create env
        self.args.N = 1  # The number of agents
        self.args.obs_dim = 5  # The dimensions of an agent's observation space
        self.args.action_dim = 2  # The dimensions of an agent's action space
        self.args.state_dim = 5  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)
        self.env = CustomEnvironment()

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
        evaluate_num = 0  # Record the number of evaluations
        p = np.empty((0, 3))
        self.win_num = 0
        while self.total_steps < self.args.max_train_steps:

            # #self.agent_n.actor.load_state_dict(torch.load("./model_actor-0.1+0.1_3e4/step_2995000.pth"))  #在这里导入旧模型
            # if self.total_steps % self.args.save_interval == 0:
            #     self.save_models()


            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                rn = self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                win_rate = self.win_num / 5
                evaluate_num += 1
                pp = np.array([[self.total_steps, rn, win_rate]]);
                print("evaluate_num:{}\t total_steps:{} \t evaluate_reward:{} \t win_rate:{}".format(evaluate_num,
                                                                                                     self.total_steps,
                                                                                                     rn, win_rate))
                p = np.vstack((p, pp))
                # file_name = "20-(2e4)_seed{}-.txt".format(seed_)
                file_name = "20-(2e4)_seed5.txt"
                np.savetxt(file_name, p, fmt='%f', delimiter=',')

                self.win_num = 0

            episode_reward, episode_steps, self.w,r = self.run_episode_mpe(evaluate=False)  # Run an episode

            self.total_steps += self.args.episode_limit
            self.win_num += self.w
            if self.replay_buffer.episode_num == self.args.batch_size:

                 self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                 self.replay_buffer.reset_buffer()

        #self.evaluate_policy()
        #self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _, _, r = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times

        self.evaluate_rewards.append(evaluate_reward)
        torch.save(self.agent_n.actor.state_dict(),
                   "./model_actor/step_{}.pth".format(self.total_steps))
        torch.save(self.agent_n.critic.state_dict(),
                   "./model_critic/step_{}.pth".format(self.total_steps))
        return evaluate_reward


    def run_episode_mpe(self, evaluate=False):
        # def generate_xt_zt():
        #     xt = 4800  # np.random.randint(4000, 5000)
        #     zt = 8000  # np.random.randint(7000, 8000)
        #     return xt, zt
        #
        # xt, zt = generate_xt_zt()
        #
        # # 定义预测函数
        # def predict_turn_start(xt, zt):
        #     # 加载模型
        #     loaded_model = joblib.load('saved_model.pkl')
        #     # 准备输入数据
        #     target_random = np.array([[xt, zt]])
        #     # 使用加载的模型进行预测
        #     turn_start = 11.4  # loaded_model.predict(target_random)
        #     return turn_start

        turn_start = 8
        # print(xt,zt,turn_start)

        RT_0 = np.array([3800, 5000, 1200])
        # RT_0 = np.array([2000, 1500, 100])
        VT_0 = 0.6 * 343
        psi_t_0 = 0
        gma_t_0 = 0
        last_ny = 0
        last_nz = 0
        target_state_0 = np.array([*RT_0, VT_0, psi_t_0, gma_t_0])

        # missile starts at the same spot with the same velocity
        # RM_0= np.random.uniform(0, 1000, 3)
        RM_0 = np.array([0, 2000, 0])

        VM_0 = 1.2 * 343
        engage_params = self.env.acquisition(RT_0, RM_0, last_ny, last_nz)
        psi_m_0 = engage_params[0]
        gma_m_0 = engage_params[1]

        t = 0
        dt = 0.01
        ny = 0
        nz = 0

        missile_state_0 = np.array([*RM_0, VM_0, psi_m_0, gma_m_0, last_ny, last_nz])
        seeker_state_0 = self.env.get_seeker_state(target_state_0, missile_state_0)
        tgo_0 = self.env.get_tgo(seeker_state_0, ny, nz)

        missile_state = missile_state_0
        target_state = target_state_0
        t_sim = [t]
        missile_sim = np.array(missile_state)  # ACTUAL MISSILE STATE
        target_sim = np.array(target_state)  # ACTUAL TARGET STATE
        distance = np.linalg.norm(missile_state[:3] - target_state[:3])

        distance_sim = [distance]

        seeker_state = seeker_state_0
        tgo = tgo_0
        eps = 6.

        # obs_n =seeker_state
        obs_n = [seeker_state[0], seeker_state[1], seeker_state[2], distance, tgo]

        episode_reward = 0
        distancel = 0
        p = 0
        self.w = 0
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        min_distance = float('inf')  # 初始化最小距离为无穷大

        for episode_step in range(self.args.episode_limit):
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)
            # a_n, a_logprob_n = self.agent_n.choose_action(obs_n)
            s = np.array(obs_n).flatten()  # In MPE, global state is the concatenation of all agents' local obs.
            v_n = self.agent_n.get_value(s)
            # last_ny, last_nz, ny, nz = self.env.pn_guidance(seeker_state, a_n, last_ny, last_nz, dt, turn_start)
            last_ny, last_nz, ny, nz = self.env.pn_guidance(seeker_state, missile_state, a_n, last_ny, last_nz, dt, turn_start)  # 增加弹道倾角信息(L
            tgo = self.env.get_tgo(seeker_state, ny, nz)
            missile_state = self.env.get_missile_state(ny, nz, missile_state, dt)
            target_state = self.env.get_target_state(dt, t, target_state, turn_start)
            seeker_state = self.env.get_seeker_state(target_state, missile_state)
            distance = np.linalg.norm(missile_state[:3] - target_state[:3])

            # 更新最小距离
            if distance < min_distance:
                min_distance = distance

            t = t + dt
            t_sim.append(t)
            missile_sim = np.vstack((missile_sim, missile_state))
            target_sim = np.vstack((target_sim, target_state))
            distance_sim.append(distance)

            Vc_track = -(seeker_state[0] * seeker_state[3] + seeker_state[1] * seeker_state[4] + seeker_state[2] *
                         seeker_state[5]) / np.linalg.norm(seeker_state[:3])

            # r_n = get_reward1(missile_state[:3], target_state[:3], r0)
            r_n = self.env.get_reward(missile_state[:3], target_state[:3], distance,tgo,eps)

            # print(r_n)

            # p += r_n

            episode_reward += r_n

            # if p > episode_reward:
            #     episode_reward = p
            #     ti = episode_step

            if distance < eps:
                done_n = True
                self.w += 1
                print("win",[distance],f"击中时间：{t:.2f} 秒")
                # print([distance])
                # print(f"击中时间：{t:.2f} 秒")  # 输出击中时间，保留两位小数
            else:
                done_n = False
                # print("miss")
                # print([distance])

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transition
                '''
                if done_n:
                    self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, episode_reward,
                                                        done_n)
                else:
                    self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)
                '''

                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = [seeker_state[0], seeker_state[1], seeker_state[2], distance, tgo]

            if done_n:
                break

        if not evaluate:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 创建三维坐标

            ax.plot3D(target_sim[:, 0], target_sim[:, 2], target_sim[:, 1],
                      linewidth=1.5, linestyle='-.', color='maroon',
                      label='target')

            # 绘制导弹1的轨迹
            ax.plot3D(missile_sim[:, 0], missile_sim[:, 2], missile_sim[:, 1],
                      linewidth=1.5, linestyle='-.', color='blue',
                      label='missile1')

            # 设置坐标轴标签
            ax.set_xlabel("X 轴")
            ax.set_ylabel("Y 轴")
            ax.set_zlabel("Z 轴")
            # 设置图例
            ax.legend()
            # 显示图表
            # plt.show()
            # plt.pause(0.5)  # 显示秒数
            # plt.close()

            missile_x = missile_sim[:, 0]
            missile_y = missile_sim[:, 1]
            missile_z = missile_sim[:, 2]
            target_x = target_sim[:, 0]
            target_y = target_sim[:, 1]
            target_z = target_sim[:, 2]

            # 保存数据为 .mat 文件
            scipy.io.savemat('missile_sim_x.mat', {'data': missile_x})
            scipy.io.savemat('missile_sim_y.mat', {'data': missile_y})
            scipy.io.savemat('missile_sim_z.mat', {'data': missile_z})
            scipy.io.savemat('target_sim_x.mat', {'data': target_x})
            scipy.io.savemat('target_sim_y.mat', {'data': target_y})
            scipy.io.savemat('target_sim_z.mat', {'data': target_z})

            # An episode is over, store v_n in the last step
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

            if done_n:
                self.replay_buffer.tianjiap2(episode_step, self.args.episode_limit - episode_step, r_n, True)
            else:
                self.replay_buffer.tianjiap2(episode_step, self.args.episode_limit - episode_step, r_n, True)

            # 如果未击中目标，输出最小距离
            if not done_n:
                print("miss", [min_distance])

        return episode_reward, self.args.episode_limit, self.w, distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(4e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=2000, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=10000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--algorithm", type=str, default="QMIX", help="QMIX or VDN")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=2e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=2e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
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
    # seed = [5]
    # for i in range(len(seed)):
    #     seed_ = seed[i]
    #     runner = Runner_MAPPO_MPE(args, env_name="missile_nav_testing", number=1, seed=seed_)
    #     runner.run()
    runner = Runner_MAPPO_MPE(args, env_name="missile_nav_testing", number=1, seed=5)
    runner.run()

print("pass")
