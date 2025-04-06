import numpy as np
import random
import gym

from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
#from  files import MultiPro
from scripts.agent import Agent
from  scripts import MultiPro
import json
from sumo_env import SumoEnv
from pandas import DataFrame
def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def evaluate(frame, eval_runs=5, capture=False, render=False):
    """
    Makes an evaluation run
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        if render: eval_env.render()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0))
            action_v = np.clip(action, action_low, action_high)

            state, reward, done, _ = eval_env.step(action_v[0])
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    if capture == False:
        writer.add_scalar("Reward", np.mean(reward_batch), frame)

def writenewtrips():
    edges=['VM M0 M1 M2 M3 M4','VR R0 R1 R2 M3 M4']#设置路径
    m1flow = np.round(np.array([799,808,811,817,809,812,807,797]))  # 主线流量
    r1flow = np.round(np.array([142,140,150,184,195,156,162,189]))  # 匝道流量
   # m1flow = np.round(np.array([699, 708, 757, 817, 809, 812, 807, 797]))  # 主线流量
    # r1flow = np.round(np.array([122, 120, 164, 184, 195, 156, 162, 189]))  # 匝道流量
    #r1flow = np.round(np.array([122, 120, 164, 260, 250, 250, 250, 230]))  # 匝道流量
    #m1flow = np.round(np.array([300, 300, 300, 800, 800, 800, 800, 800]))  # 主线流量
    # r1flow = np.round(np.array([122, 120, 164, 184, 195, 156, 162, 189]))  # 匝道流量
   # r1flow = np.round(np.array([60, 60, 60, 195, 195, 195, 195, 195]))  # 匝道流量
    with open('nomergearea.sumocfg', 'w') as f:
        f.write('<configuration>\n')
        f.write('    <input>\n')
        f.write('        <net-file value="noMergeArea.net.xml"/>\n')
        f.write('        <route-files value="nomergearea.rou.xml"/>\n')
        f.write('        <additional-files value="detectors.add.xml"/>\n')
        f.write('    </input>\n')
        # 生成1到100之间的随机整数作为随机种子
        seed = random.randint(1, 100)
        f.write("""    <seed value=\"""" + str(seed) +  """\"/> """ + '\n')
        f.write('    <time>\n')
        f.write('        <begin value="0"/>\n')
        f.write('        <end value="4800"/>\n')
        f.write('    </time>\n')
        f.write('    <processing>\n')
        f.write('        <time-to-teleport value="-1"/>\n')
        f.write('        <max-depart-delay value="-1"/>\n')
        f.write('        <extrapolate-departpos value="true"/>\n')
        f.write('        <emergency-insert value="true"/>\n')
        f.write('        <collision.stoptime value="0"/>\n')
        f.write('    </processing>\n')
        f.write('</configuration>\n')
    with open('nomergearea.rou.xml', 'w') as routes:
        routes.write("""<routes>""" + '\n')
        routes.write('\n')
        #车辆属性，主要在这里调参

        routes.write("""<vType id="type2" color="yellow" vClass="passenger" length = "5" lcKeepRight="-1" sigma="0.5" tau="1.3" mingap="2.0" accel="3.0" decel="4.5" speedFactor="1.0" speedDev="0.04" lcAssertive="3.5" lcStrategic="0.5" lcSpeedGain="1.0" lcImpatience="0.5"  lcCooperative="1.0" lcLookaheadLeft="5.0"/>"""+ '\n')
        #routes.write(
       #     """<vType id="type2" color="yellow" vClass="passenger" length = "5" lcKeepRight="-1" sigma="0.5" tau="1.3" mingap="2.0" accel="3.0" decel="4.5" speedFactor="1.0" speedDev="0.04" lcAssertive="3.5" lcStrategic="0.5" lcSpeedGain="5.0" lcImpatience="0.5"  lcCooperative="1.0" lcLookaheadLeft="5.0"/>""" + '\n')
       # routes.write('\n')
        for i in range(len(edges)):
            routes.write("""<route id=\"""" + str(i) + """\"""" + """ edges=\"""" + edges[i] + """\"/> """ + '\n')
        temp = 0
        #np.random.seed(1)
        for t in range(len(m1flow)):
            m_in = int(m1flow[t]) + np.random.normal(loc=0, scale=40)
            r1_in = int(r1flow[t]) + np.random.normal(loc=0, scale=15)

            vNum =m_in + r1_in
            dtime = np.random.uniform(0+600*t,600+600*t,size=(int(vNum),))
            dtime.sort()
            for veh in range(int(vNum)):
                typev = np.random.choice([2,3], p = [1,0])#选择车辆类型，这里的意思是50%的概率选类型2，50%的概率选类型3
                vType = 'type' + str(typev)
                route = np.random.choice([0,1], p =[m_in/vNum,r1_in/vNum])#按路径流量分配车辆路径的比例
                #这里也可以进行部分调参
                if route==0:
                    routes.write("""<vehicle id=\"""" + str(temp+veh) + """\" depart=\"""" + str(round(dtime[veh],2)) + """\" type=\"""" + str(vType) + """\" route=\"""" + str(route) + """\" departLane=\"random"""+"""\" departPos=\"base""" +"""\" departSpeed=\"max"""+"""\"/>"""+ '\n')
                else:
                    routes.write("""<vehicle id=\"""" + str(temp+veh) + """\" depart=\"""" + str(round(dtime[veh],2)) + """\" type=\"""" + str(vType) + """\" route=\"""" + str(route) + """\" departLane=\"random"""+"""\" departPos=\"base""" +"""\" departSpeed=\"max"""+"""\"/>"""+ '\n')
                routes.write('\n')
            temp+=vNum
        routes.write("""</routes>""")

def run(frames=1000, worker=1):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=80)  # last 80 scores
    i_episode = 1
    all_ep_r = []
    for i in range(worker+1):
         all_ep_r.append([])
    state = envs.reset()
    score = 0
    curiosity_logs = []
    act_dim=action_size
    DControl = [[100]*10,[100]*10,[100]*10,[60]*10]
    act_set=[]
    best_score=float('-inf')
    for frame in range(1, frames+1):

        # evaluation runs
        #if frame % eval_every == 0 or frame == 1:
        #    evaluate(frame*worker, eval_runs)
        action_v = agent.act(state) * action_high
        var = action_high * 0.3
        #action_v_rl = agent.act(state)
        #var = action_high * 0.2
        #action_v = np.clip(np.random.normal(action_v_rl* action_high, var), -action_high, action_high, dtype='float32')
        action_v = np.clip(np.random.normal(action_v, var), -action_high, action_high, dtype='float32')
        next_state, reward, done, real_act = envs.step(action_v)
        act_set.append(real_act)

        #print(next_state,reward,done)
        for s, a, r, ns, d in zip(state, action_v, reward, next_state, done):
        #for s, a, r, ns, d in zip(state, action_v_rl, reward, next_state, done):
            agent.step(s, a, r, ns, d, frame, writer)
            
        if args.icm:
            reward_i = agent.icm.get_intrinsic_reward(state[0], next_state[0], action_v[0])
            curiosity_logs.append((frame, reward_i))
        state = next_state
        score += reward
        if done.any():

            scores_window.append(score)       # save most recent score
            scores.append(score)    # save most recent score

            all_ep_r[0].append(np.mean(score))
            data={'AVG_Reward':all_ep_r[0]}
            for i in range(1,worker+1):
                all_ep_r[i].append(score[i-1])
                data['Reward_'+str(i)]=all_ep_r[i]

            df = DataFrame(data)
            df.to_excel('Nomerge_PF_NC_Reward.xlsx')
            # 找到最大元素的索引

            max_index = np.argmax(score)
            if max(score)>best_score:
                best_score=max(score)
                for real in act_set:
                    best_act=real[max_index]
                    for index, act in enumerate(best_act):
                        DControl[index].append(act)
                d={'M1_0':DControl[0],'M1_1':DControl[1],'M1_2':DControl[2],'RM':DControl[3]}
                df1 = DataFrame(d)
                df1.to_excel('PF_NC.xlsx')
                """
                # save trained model
                torch.save(agent.actor_local.state_dict(), 'runs/' + args.info + '_actor_local' + ".pth")
                torch.save(agent.actor_target.state_dict(), 'runs/' + args.info + '_actor_target' + ".pth")
                torch.save(agent.critic_local.state_dict(), 'runs/' + args.info + '_critic_local' + ".pth")
                torch.save(agent.critic_target.state_dict(), 'runs/' + args.info + '_critic_target' + ".pth")
                # save parameter
                with open('runs/' + args.info + ".json", 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
                    """
            writer.add_scalar("Average80", np.mean(scores_window), frame*worker)
            for v in curiosity_logs:
                i, r = v[0], v[1]
                writer.add_scalar("Intrinsic Reward", r, i)
            print('\rEpisode {}\tFrame {} \tAverage80 Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            #if i_episode % 100 == 0:
            #    print('\rEpisode {}\tFrame \tReward: {}\tAverage100 Score: {:.2f}'.format(i_episode*worker, frame*worker, round(eval_reward,2), np.mean(scores_window)), end="", flush=True)
            i_episode +=1
            writenewtrips()
            state = envs.reset()
            DControl = [[100]*10,[100]*10,[100]*10,[60]*10]
            act_set=[]
            score = 0
            curiosity_logs = []
            




parser = argparse.ArgumentParser(description="")
#parser.add_argument("-env", type=str,default="HalfCheetahBulletEnv-v0", help="Environment name, default = HalfCheetahBulletEnv-v0")
parser.add_argument("--device", type=str, default="gpu", help="Select trainig device [gpu/cpu], default = gpu")
parser.add_argument("-nstep", type=int, default=1, help ="Nstep bootstrapping, default 1")#不可调
parser.add_argument("-per", type=int, default=1, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")#不可调
parser.add_argument("-munchausen", type=int, default=0, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 0")#不可调
parser.add_argument("-iqn", type=int, choices=[0,1], default=1, help="Use distributional IQN Critic if set to 1, default = 1")#不可调
parser.add_argument("-noise", type=str, choices=["ou", "gauss"], default="gauss", help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")#不可调
parser.add_argument("-info", type=str,default="test8", help="Information or name of the run")#不可调
parser.add_argument("-d2rl", type=int, choices=[0,1], default=0, help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")#不可调
parser.add_argument("-frames", type=int, default=1_28_00000, help="The amount of training interactions with the environment, default is 1mio")
parser.add_argument("-seed", type=int, default=50, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr_a", type=float, default=5e-4, help="Actor learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-lr_c", type=float, default=6e-4, help="Critic learning rate of adapting the network weights, default is 3e-4")
parser.add_argument("-learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("-learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("-layer_size", type=int, default=128, help="Number of nodes per neural network layer, default is 256")#不可调
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=5e-4, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor gamma, default is 0.99")
parser.add_argument("-w", "--worker", type=int, default=2, help="Number of parallel environments, default = 1")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")#不可调
parser.add_argument("--icm", type=int, default=0, choices=[0,1], help="Using Intrinsic Curiosity Module, default=0 (NO!)")#不可调
parser.add_argument("--add_ir", type=int, default=0, choices=[0,1], help="Add intrisic reward to the extrinsic reward, default = 0 (NO!) ")#不可调

args = parser.parse_args()

if __name__ == "__main__":

    seed = args.seed
    frames = args.frames
    worker = args.worker
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size * args.worker
    LR_ACTOR = args.lr_a         # learning rate of the actor 
    LR_CRITIC = args.lr_c        # learning rate of the critic
    saved_model = args.saved_model
    D2RL = args.d2rl

    writer = SummaryWriter("runs/"+args.info)
    env=SumoEnv()
    envs = MultiPro.SubprocVecEnv([lambda: env for i in range(args.worker)])
    envs.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda:0") 
    else:
        "CUDA is not available"
        device = torch.device("cpu")
    print("Using device: {}".format(device))
    
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = Agent(state_size=state_size, action_size=action_size, n_step=args.nstep, per=args.per, munchausen=args.munchausen,distributional=args.iqn,
                 D2RL=D2RL, curiosity=(args.icm, args.add_ir), noise_type=args.noise, random_seed=seed, hidden_size=HIDDEN_SIZE, BATCH_SIZE=BATCH_SIZE, BUFFER_SIZE=BUFFER_SIZE, GAMMA=GAMMA,
                 LR_ACTOR=LR_ACTOR, LR_CRITIC=LR_CRITIC, TAU=TAU, LEARN_EVERY=args.learn_every, LEARN_NUMBER=args.learn_number, device=device, frames=args.frames, worker=args.worker) 
    
    t0 = time.time()
    """
       agent.actor_local.load_state_dict(torch.load('C:/Users/HZ/Desktop/D4PG_feedback/runs/test8_actor_local.pth'))
       agent.actor_target.load_state_dict(torch.load('C:/Users/HZ/Desktop/D4PG_feedback/runs/test8_actor_target.pth'))
       agent.critic_local.load_state_dict(torch.load('C:/Users/HZ/Desktop/D4PG_feedback/runs/test8_critic_local.pth'))
       agent.critic_target.load_state_dict(torch.load('C:/Users/HZ/Desktop/D4PG_feedback/runs/test8_critic_target.pth'))
       """
    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        eval_env=SumoEnv()
        eval_env.seed(seed+1)
        evaluate(frame=None, capture=False)
    else:
        run(frames = args.frames//args.worker,
            worker=args.worker)
    t1 = time.time()
    timer(t0, t1)
    # save trained model 
    torch.save(agent.actor_local.state_dict(), 'runs/'+args.info+".pth")
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)