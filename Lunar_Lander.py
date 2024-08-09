# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:25:42 2023

@author: zhaodf
"""
#Lunar Lander环境具有一个从视窗顶部释放的着陆器，智能体需要学习如何使其安全着陆于月面
#智能体输入为着陆器的x和y坐标、x和y的线速度、角度、角速度，以及代表每条腿是否与地面接触的两个布尔运算符
#智能体可选操作为启动左引擎，启动主引擎，启动右引擎和不操作
#初始状态着陆器处于窗口上方中心，开始时会被施加一个随机的力
#回合结束条件为着陆器坠毁(主体接触月球表面)或飞出视窗
#奖励：从顶部移动到着陆点奖励约100~140分，若着陆器离开着陆台，奖励移除
#      着陆坠毁-100分，安全着陆+100分，每条腿着陆+10分
#      开启主引擎，-0.3分/step; 开启侧引擎，-0.03分/step

#详细介绍：https://gymnasium.farama.org/environments/box2d/lunar_lander/

#Tensorflow1.13.1 and Keras2.2.4

import gym
from keras.models import Model
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
# from rl.callbacks import ModelIntervalCheckpoint, FileLogger

def Lunar_Lander_model(state_size, num_actions):
    input = Input(shape=(1,state_size))
    x = Flatten()(input)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(num_actions, activation='linear')(x)
    model = Model(inputs=input, outputs=output)
    return model

# def Lunar_Lander_callbacks(env_name):
#     checkpoint_weights_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
#     log_filename = 'dqn_{}_log.json'.format(env_name)
#     callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
#     callbacks += [FileLogger(log_filename, interval=100)]
#     return callbacks

def main():
    ENV_Name = 'LunarLander-v2'
    env = gym.make(ENV_Name)
    num_actions = env.action_space.n
    state_space = env.observation_space.shape[0]
 
    model = Lunar_Lander_model(state_space, num_actions)
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, 
                                  value_min=0.1, value_test=0.05, nb_steps=10000)
    dqn = DQNAgent(model=model, nb_actions=num_actions, memory=memory, 
                   nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
    
    dqn.compile(Adam(lr=0.0003), metrics=['mae'])
    # callbacks = Lunar_Lander_callbacks(ENV_Name)
    dqn.fit(env, nb_steps=500000, visualize=True, verbose=2)
    
        # dqn.fit(env, nb_steps=500000,
        #     visualize=True,
        #     verbose=2,
        #     callbacks=callbacks)

    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_Name), overwrite=True)

    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    main()

