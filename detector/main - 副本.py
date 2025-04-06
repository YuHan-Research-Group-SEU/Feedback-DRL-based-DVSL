import numpy as np
import random
import sys
import time
import os
import pandas as pd
from ppo_torch import Agent
from utils import plot_learning_curve
from networks import rm_vsl_co
from pandas import DataFrame
tools = 'D:/Sumo/tools'
sys.path.append(tools)
def from_a_to_mlv(a):
    return [15 + np.floor(a)]
if __name__ == '__main__':
    env = rm_vsl_co(visualization = False)
    N =64
    batch_size =16
    n_epochs = 4
    alpha = 0.0005
    agent = Agent(batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs, 
                    input_dims=30,out_dim=5)
    n_games =1000
    DControl=[[29.06]*20,[29.06]*20,[29.06]*20,[30]*20]
    figure_file = './learning_curve.png'

    best_score = float('-inf')
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    warmup_time=600
    for i in range(n_games):
        Control=[30]*20
        observation = env.reset(warmup_time)
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            print(action)
            act = from_a_to_mlv(action)
            act=[29.06,29.06,29.06,act[0]]
            print(act)
            for i,t in enumerate(act):
                DControl[i].append(t)
            observation_,reward, done,out_flow_temp, bspeed_temp, co_temp, hc_temp, nox_temp, pmx_temp,done= env.step(act)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        env.close()
        score_history.append(score)
        data={'Episode_Reward':score_history}
        df = DataFrame(data)
        df.to_excel('Nomerge_R_Reward.xlsx')
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        if score==max(score_history):
            d={'V0':DControl[0],'V1':DControl[1],'V2':DControl[2],'Ramp_act':DControl[3]}
            df = DataFrame(d)
            df.to_excel('Nomerge_R.xlsx')
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


