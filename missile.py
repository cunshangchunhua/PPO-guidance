import math

import numpy as np
from math import sin, cos, atan, asin, sqrt


class CustomEnvironment(object):
    def __init__(self, ):
        self.x0 = 0
        self.z0 = 0

        pos_noise_std = 20
        vel_noise_std = 20

        # 定义噪声参数
        self.pos_noise_std = pos_noise_std  # 位置测量噪声标准差（单位：米）
        self.vel_noise_std = vel_noise_std  # 速度测量噪声标准差（单位：米/秒）

    def acquisition(self, target_pos, missile_pos, last_ny, last_nz):
        """
        Sets initial heading and climb angle to LOS angles, better ensuring a collision
        :param target_pos: initial detected position of target
        :return: recommended initial climb angle and heading angle
        """

        # noisy_target_pos = [
        #     target_pos[0] + np.random.normal(0, self.pos_noise_std),
        #     target_pos[1] + np.random.normal(0, self.pos_noise_std),
        #     target_pos[2] + np.random.normal(0, self.pos_noise_std)
        # ]

        Rx = target_pos[0] - missile_pos[0]
        Ry = target_pos[1] - missile_pos[1]
        Rz = target_pos[2] - missile_pos[2]

        Rxz = np.linalg.norm([Rx, Rz])
        R = np.linalg.norm(target_pos)

        psi_rec = atan(-Rz / Rx)
        gma_rec = atan(Ry / Rxz)
        return [psi_rec, gma_rec]

    def get_seeker_state(self, target_state, missile_state):
        """
        Computes the "seeker state," relative position and velocity in inertial coordinates
        :param target_state: position, speed, and orientation of target
        :param missile_state: position, speed, and orientation of missile
        :return: the seeker state, 3 coordinates of inertial position, 3 components of inertial velocity
        """
        # define variables for readability
        # T = target
        # M = missile

        # Target position
        RTx = target_state[0]
        RTy = target_state[1]
        RTz = target_state[2]

        # Target velocity and heading

        VT = target_state[3]
        psi_t = target_state[4]
        gma_t = target_state[5]

        # Missile Position
        RMx = missile_state[0]
        RMy = missile_state[1]
        RMz = missile_state[2]

        # Missile velocity and heading
        VM = missile_state[3]
        psi_m = missile_state[4]
        gma_m = missile_state[5]

        # Rotation matrices from inertial to body fixed coordinates for
        # both the target and the missile

        vxt = VT * cos(gma_t) * cos(psi_t)  # gma_t是弹道倾角 psi_t是弹道偏角
        vyt = VT * sin(gma_t)
        vzt = -VT * cos(gma_t) * sin(psi_t)
        vxm = VM * cos(gma_m) * cos(psi_m)
        vym = VM * sin(gma_m)
        vzm = -VM * cos(gma_m) * sin(psi_m)

        kr = 25
        kv = 25

        rel_pos = [
            (RTx - RMx) ,
            (RTy - RMy) ,
            (RTz - RMz)
        ]
        rel_vel = [
            (vxt - vxm),
            (vyt - vym) ,
            (vzt - vzm)
        ]
        return np.array(rel_pos + rel_vel)

        # return np.array([RTx - RMx, RTy - RMy, RTz - RMz,
        #                  vxt - vxm, vyt - vym, vzt - vzm])

    def pn_guidance(self, seeker_state, missile_state, a_n, last_ny, last_nz, dt, turn_start):
        """
        Provides commanded accelerations using proportional navigation law
        :param seeker_state: current seeker state
        :return: Commanded accelerations in inertial coordinates
        """

        Rx = seeker_state[0]
        Ry = seeker_state[1]
        Rz = seeker_state[2]
        Vx = seeker_state[3]
        Vy = seeker_state[4]
        Vz = seeker_state[5]

        beta = atan(Ry / sqrt(Rx ** 2 + Rz ** 2))  # 视线倾角
        if Rx == 0 and Rz > 0:  # % % % % %（有问题）
            alpha = -np.pi / 2
        elif Rx == 0 and Rz < 0:
            alpha = np.pi / 2
        else:
            alpha = atan(-Rz / Rx)  # 视线偏角
        R = np.linalg.norm([Rx, Ry, Rz])
        beta_dot = ((Rx ** 2 + Rz ** 2) * Vy - Ry * (Rx * Vx + Rz * Vz)) / (
                    R * R * sqrt(Rx ** 2 + Rz ** 2))  # 视线倾角变化率
        alpha_dot = (Rz * Vx - Rx * Vz) / (Rx ** 2 + Rz ** 2)  # 视线偏角变化率

        dr = (Rx * Vx + Ry * Vy + Rz * Vz) / R
        g = 9.8
        tau = 0.5

        # ny = (a_n[0] * 2 + 4) * abs(dr) * beta_dot / g
        # ny = 5 * abs(dr) * beta_dot / g + a_n[0]
        ny = (a_n[0] * 2 + 4) * abs(dr) * beta_dot / g + cos(missile_state[5]) #+ a_n[0]  # 增加重力补偿(L
        dny = (ny - last_ny) / tau
        ny = dny * dt + last_ny
        last_ny = ny

        # nz = -(a_n[1] * 2 + 4) * abs(dr) * alpha_dot /g #+ a_n[1]*100
        nz = -(a_n[0] * 2 + 4) * abs(dr) * alpha_dot / g #+ a_n[1]
        dnz = (nz - last_nz) / tau
        nz = dnz * dt + last_nz
        last_nz = nz

        if abs(ny) >= 30:
            ny = 30 * self.sign(ny)
        if abs(nz) >= 30:
            nz = 30 * self.sign(nz)

        # front_angle = math.atan2(math.sin(beta), math.cos(beta) * math.sin(alpha))
        #
        # tgo = (R/Vm)*(1+(front_angle**2/(4)))

        # tgo = sqrt(Rx**2 + ny**2*Ry**2 + nz**2*Rz**2)/abs((Vx*Rx + ny*Vy*Ry + nz*Vz*Rz)/sqrt(Rx**2 + ny**2*Ry**2 + nz**2*Rz**2))

        return last_ny, last_nz, ny, nz

    def get_tgo(self, seeker_state, ny, nz):

        Rx = seeker_state[0]
        Ry = seeker_state[1]
        Rz = seeker_state[2]
        Vx = seeker_state[3]
        Vy = seeker_state[4]
        Vz = seeker_state[5]
        tgo = sqrt(Rx ** 2 + ny ** 2 * Ry ** 2 + nz ** 2 * Rz ** 2) / abs(
            (Vx * Rx + ny * Vy * Ry + nz * Vz * Rz) / sqrt(Rx ** 2 + ny ** 2 * Ry ** 2 + nz ** 2 * Rz ** 2))
        return tgo

    def sign(self, input):
        if input > 0:
            return 1
        elif input < 0:
            return -1
        else:
            return 0

    def get_missile_state(self, ny, nz, missile_state, dt):

        g = 9.8
        dvm = 0 * g
        dvm = -g * sin(missile_state[5])  # 重力对导弹减速作用(L

        # 纯比例导引法 指令加速度与速度矢量垂直(L
        dthetam = 9.8 / missile_state[3] * (ny - cos(missile_state[5]))
        dposaim = -9.8 * nz / (missile_state[3] * cos(missile_state[5]))  # azm = g * nz
        dxm = missile_state[3] * cos(missile_state[5]) * cos(missile_state[4])
        dym = missile_state[3] * sin(missile_state[5])
        dzm = -missile_state[3] * cos(missile_state[5]) * sin(missile_state[4])
        missile_state[3] = missile_state[3] + dvm * dt
        if missile_state[3] > 1700:
            missile_state[3] = 1700

        missile_state[5] = missile_state[5] + dthetam * dt
        missile_state[4] = missile_state[4] + dposaim * dt
        missile_state[0] = missile_state[0] + dxm * dt
        missile_state[1] = missile_state[1] + dym * dt
        missile_state[2] = missile_state[2] + dzm * dt

        return missile_state

    def get_reward(self, missile_pos, target_pos, distance, tgo, eps):

        if distance > 0:
            state_rew = 1 / distance
        else:
            state_rew = 0.0

            # 终端奖励（击中目标）
        if tgo <= 0 or distance < eps:  # 假设tgo为0或者距离很小视为击中
            terminal_rew = 50.0
        else:
            terminal_rew = 0.0

            # 过程惩罚项
        if tgo > 10:
            process_rew = -1  # 惩罚长时间未完成任务
        else:
            process_rew = 2

        if tgo > 0:
            approach_rew = 1 / (tgo ** 1)  # 更强调时间较长的情况下的接近
        else:
            approach_rew = 0.0

            # 总奖励计算
        rew = (state_rew * 1.0) + (process_rew * 0.8) + (approach_rew * 1.0) + (terminal_rew * 1.0)

        return rew

    # def get_target_state(self, dt, t, target_state, turn_start):
    #     g = 12  #转弯过载
    #     turn_radius = target_state[3]**2/9.8/g  #转弯半径
    #     #turn_start = 3.8   #转弯时间
    #     turn_duration = 2  # 转弯持续的时间
    #     angularSpeed = target_state[3] / turn_radius   #角速度
    #
    #     if t < turn_start:   #  第一段直线飞行
    #         target_state[0] = target_state[0]
    #         target_state[2] = target_state[2]+ target_state[3] * dt
    #         self.x0 = target_state[0]
    #         self.z0 = target_state[2]
    #         target_state[4] = -np.pi/2
    #     elif t >= turn_start and t <= (turn_start + turn_duration):
    #
    #         tg = t - turn_start
    #         target_state[4] =  -np.pi/2+angularSpeed * tg
    #         target_state[0] = self.x0 - turn_radius * cos(angularSpeed * tg)+ turn_radius
    #         target_state[2] = self.z0 + turn_radius * sin(angularSpeed * tg)
    #     else:
    #         # % 第三段直线飞行
    #         # % 调整方向为转弯后的方向
    #         target_state[0] = target_state[0] + target_state[3] * cos(target_state[4]) * dt
    #         target_state[2] = target_state[2] - target_state[3] * sin(target_state[4]) * dt
    #
    #     target_state = np.array(
    #         [target_state[0], target_state[1], target_state[2], target_state[3], target_state[4], target_state[5]])   #target[4]是弹道偏角
    #
    #     return target_state

    def get_target_state(self, dt, t, target_state, turn_start):
        g = 12  # 转弯过载
        turn_radius = target_state[3] ** 2 / 9.8 / g  # 转弯半径
        # turn_start = 12 # 转弯开始时间
        angularSpeed = target_state[3] / turn_radius  # 角速度
        s_freq = 0.2  # S形频率

        if t < turn_start:  # 第一段直线飞行
            target_state[0] = target_state[0]
            target_state[2] = target_state[2] + target_state[3] * dt
            self.x0 = target_state[0]
            self.z0 = target_state[2]
            target_state[4] = -np.pi / 2

        else:  # 开始持续S形机动
            # 计算当前应该转向哪个方向
            turn_direction = np.sin(2 * np.pi * s_freq * (t - turn_start))

            # 计算当前航向角的变化
            delta_heading = angularSpeed * dt * turn_direction
            target_state[4] += delta_heading

            # 根据当前航向角更新位置
            target_state[0] = target_state[0] + target_state[3] * cos(target_state[4]) * dt
            target_state[2] = target_state[2] - target_state[3] * sin(target_state[4]) * dt

        target_state = np.array(
            [target_state[0], target_state[1], target_state[2], target_state[3], target_state[4], target_state[5]])

        return target_state
