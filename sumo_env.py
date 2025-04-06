import time

import numpy as np
from gym import spaces
from gym import Env
import os
import sys
import random
tools ='D:/SUMO/sumo-win64-1.17.0/sumo-1.17.0/tools' #改为你sumo的tools地址即可
sys.path.append(tools)
from collections import defaultdict
import traci
import math
#from module.CTM_network_VSL import CTM_network
sumoConfig = "nomergearea.sumocfg"

class SumoEnv:
    def __init__(self,visualization = False, control_horizon =60):

        self.control_section = 'M1'
        self.state_detector = ['e1_0','e1_1','e1_2','M1_0','M1_1','M1_2','e1_12','e1_13','e1_14','e1_rin0','e1_rin1','R0_0','R0_1','R1_1','R2_1','e1_rout','M2_0','M2_1','M2_2','e1_VM0','e1_VM1','e1_VM2','e1_24','e1_25','e1_26']
        self.VSLlist = ['M1_0','M1_1','M1_2']
        self.discharge = ['M2_0','M2_1','M2_2']
        self.Ramp_list=['R1_1','R2_1']
        self.inID = ['e1_VM0','e1_VM1','e1_VM2','e1_rin0','e1_rin1']
        self.outID = ['e1_24','e1_25','e1_26']
        self.bottleneck_detector = ['e1_15','e1_16','e1_17']
        self.total_detector=['M0','M1','M2','M3','M4','R0','R1','R2']
        self.que_detector=['e2_0','e2_1']
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(39,))
        self.action_space = spaces.Box(low=-15, high=15, shape=(3,))# -10 10
        self.seedvalue=1
        '''
        Feedback Control Parameters
        '''
        #主线消散路段元胞的相关参数
        self.Cu=[1500,1800,1800]
        self.Cd=[1500,1800,1800]
        self.vf=[72,72,72]
        self.pc1=[20.8,25,25]#原先的
        self.pc=[20,25,25]#18
        self.pj=[125,125,125]
        self.pj_=[140,140,140]
        self.w=[14.3,18,18]
        self.w_=[round(self.Cu[0]/(self.pj_[0]-self.pc1[0]),2),round(self.Cu[1]/(self.pj_[1]-self.pc1[1]),2),round(self.Cu[2]/(self.pj_[2]-self.pc1[2]),2)]
        '''''
        self.Cu = [1500, 2050, 2050]
        self.Cd = [1500, 2050, 2050]
        self.vf = [90, 90, 90]
        self.pc1 = [16.67, 22.78, 22.78]  # 原先的
        self.pc = [16.67, 22.78, 22.78]  # 18
        self.pj = [140, 140, 140]
        self.pj_ = [150, 150, 150]
        self.w = [12.16, 17.49, 17.49]
        self.w_ = [round(self.Cu[0] / (self.pj_[0] - self.pc1[0]), 2),
                   round(self.Cu[1] / (self.pj_[1] - self.pc1[1]), 2),
                   round(self.Cu[2] / (self.pj_[2] - self.pc1[2]), 2)]
        '''''
        #匝道相关参数
        self.vf_r=[50]
        self.C_r=[1200]
        self.pc_r=[24]
        self.pj_r=[140]
        self.pj__r=[150]
        self.w_r=[10.34]
        self.w__r=[round(self.C_r[0]/(self.pj__r[0]-self.pc_r[0]),2)]
        self.u=[110,110,110]
        self.merge_rate0=1200
        self.length=0.25#消散路段长度
        self.r=1200
        self.greentime=[60]
        self.redtime=50


        '''
        Simulation Parameters
        '''
        self.simulation_hour = 1.5  #hours
        self.simulation_step = 0
        self.control_horizon = control_horizon  #seoncs
        self.visualization = visualization
        if self.visualization == False:
            self.sumoBinary = "D:/SUMO/sumo-win64-1.17.0/sumo-1.17.0/bin/sumo"#改为你sumo的/bin/sumo地址即可
        else:
            self.sumoBinary = "D:/SUMO/sumo-win64-1.17.0/sumo-1.17.0/bin/sumo-gui"#改为你sumo的/bin/sumo-gui地址即可


     #####################  obtain state  #################### 
    def get_step_state(self):
        state = []
        for detector in self.state_detector:
            if detector[:2]=='e1':
                veh_count= traci.inductionloop.getLastStepVehicleNumber(detector)
                if veh_count<0:
                    veh_count = 0
                state.append(veh_count)
            else:
                occupancy=traci.lane.getLastStepOccupancy(detector)
                if occupancy<0:
                    occupancy=0
                state.append(occupancy)
                avgspeed=traci.lane.getLastStepMeanSpeed(detector)
                if avgspeed<0:
                    avgspeed=27.78
                state.append(avgspeed/10.00)
        return np.array(state)
    
    #####################  set speed limit  #################### 
    def set_vsl(self, act):
        number_of_lane =3
        for j in range(number_of_lane):
            traci.lane.setMaxSpeed(self.VSLlist[j], act[j])
            
    #####################  the out flow ####################         
    def calc_outflow(self):
        '''state_out = []
        state_in = []
        for detector in self.outID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            if veh_num<0:
                veh_num=0
            state_out.append(veh_num)
        for detector in self.inID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            if veh_num<0:
                veh_num=0
            state_in.append(veh_num)
        #return np.sum(np.array(state_out)) - np.sum(np.array(state_in))'''
        arrival_num=0
        if traci.simulation.getArrivedNumber()>=0:
            arrival_num=traci.simulation.getArrivedNumber()
        return arrival_num
    
    #####################  the bottleneck speed ####################  
    def calc_bottlespeed(self):
        speed = []
        for detector in self.bottleneck_detector:
            dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
            if dspeed>=0:
                speed.append(dspeed)
        if speed:
            return np.mean(speed)
        else:
            return 33.33
    #####################  the waiting time #################### 
    def calc_waitingtime(self):
        vidlist =['m2','m3','m4','in0']
        waitingtime= []
        for vid in vidlist:
            halttime=traci.edge.getWaitingTime(vid)
            if halttime<0:
                halttime=0
            waitingtime.append(halttime)
        return np.sum(np.array(waitingtime))
    #####################  the Emergency count #################### 
    def calc_emergency(self):
        # vidlist =['m2','m3','m4','m5','m6','m7','m8','m9','m10','m11']
        vidlist =traci.edge.getLastStepVehicleIDs('m2')+traci.edge.getLastStepVehicleIDs('m3')+traci.edge.getLastStepVehicleIDs('m4')
        count=0
        for vid in vidlist:
            if traci.vehicle.getAcceleration(vid)<=-4.5:
                count+=1
        return count
    #####################  the CO, NOx, HC, PMx emission  #################### 
    
    def calc_emission(self):
        vidlist =['M1','M2']
        co = []
        hc = []
        nox = []
        pmx = []
        for vid in vidlist:
            co.append(traci.edge.getCOEmission(vid))
            hc.append(traci.edge.getHCEmission(vid))
            nox.append(traci.edge.getNOxEmission(vid))
            pmx.append(traci.edge.getPMxEmission(vid))
        return np.sum(np.array(co)),np.sum(np.array(hc)),np.sum(np.array(nox)),np.sum(np.array(pmx))
    #计算反馈控制产生的限速值
    def Feedback_ACT(self,rl_act):
        u = [100, 100, 100]
        # u = self.u+rl_act[:3]
        u[0] = self.u[0] + 4*rl_act[0]
        u[1] = self.u[1] + 4*rl_act[1]
        u[2] = self.u[2] + 4 * rl_act[2]
        #rp = np.mean(self.density[0][-self.control_horizon:])
        #self.r = self.merge_rate0 + (60 + 1.5 * rl_act[-1]) * (self.pc[0] - rp)
        #self.r = np.clip(200, self.r, 1200)
        #self.greentime[0] = np.clip(10, self.r * 30 / 1200, 30)
        #self.greentime[0] = 15+np.round(rl_act[-1])
        self.merge_rate0 = self.r
        if self.simulation_step==self.warmup_time:
            p0=[0 for _ in range(len(self.discharge))]
            qout=[0 for _ in range(len(self.discharge))]
            p0_R=np.mean(self.density_R[-self.control_horizon:])

            #for index,lane_id in enumerate(self.discharge):
            for index in range(len(self.discharge)):
                p0[index]=np.mean(self.density[index][-self.control_horizon:])
                if index==0:#靠近匝道的主线车道的索引号
                    m_demand=min(self.vf[index]*p0[index],self.w_[index]*(self.pj_[index]-p0[index]))
                    r_demand=min(self.vf_r[0]*p0_R,self.w__r[0]*(self.pj__r[0]-p0_R))
                    assert m_demand+r_demand>0,"Error! m_demand+r_demand must be postive value."
                    qout[index]=min(m_demand,max(self.Cd[index]*m_demand/(m_demand+r_demand),self.Cd[index]-self.C_r[0]))#视情况修改
                    #qout[index] = min(m_demand, self.Cd[index])# 视情况修改
                    #qin0 = qout[index] - u[index] * (p0[index] - self.pc[index]) - self.r# 视情况修改
                    qin0=qout[index]-u[index]*(p0[index]-self.pc[index])
                    qvsl=np.median((0,qin0,self.vf[index]*self.pc[index]))
                    self.feedback_act[index]=self.w[index]*qvsl/((self.w[index]*self.pj[index]-qvsl)*3.6)
                else:
                    qout[index]=min(self.vf[index]*p0[index],self.w_[index]*(self.pj_[index]-p0[index]),self.Cd[index])
                    qin0=qout[index]-u[index]*(p0[index]-self.pc[index])
                    qvsl=np.median((0,qin0,self.vf[index]*self.pc[index]))
                    self.feedback_act[index]=self.w[index]*qvsl/((self.w[index]*self.pj[index]-qvsl)*3.6)

        else:
            p=[0 for _ in range(len(self.discharge))]
            p_R=np.mean(self.density_R[-self.control_horizon:])
            qout=[0 for _ in range(len(self.discharge))]
            #for index,lane_id in enumerate(self.discharge):
            for index in range(len(self.discharge)):
                p[index]=np.mean(self.density[index][-self.control_horizon:])
                print(index,p[index])
                if index==0:#靠近匝道的主线车道的索引号
                    m_demand=min(self.vf[index]*p[index],self.w_[index]*(self.pj_[index]-p[index]))
                    r_demand=min(self.vf_r[0]*p_R,self.w__r[0]*(self.pj__r[0]-p_R))
                    assert m_demand+r_demand>0,"Error! m_demand+r_demand must be postive value."
                    qout[index]=min(m_demand,max(self.Cd[index]*m_demand/(m_demand+r_demand),self.Cd[index]-self.C_r[0]))
                    #qout[index] = min(m_demand, self.Cd[index])# 视情况修改
                    qin0=qout[index]-u[index]*(p[index]-self.pc[index])
                    #qin0 = qout[index] - u[index] * (p[index] - self.pc[index])-self.r
                    qvsl=np.median((0,qin0,self.vf[index]*self.pc[index]))
                    self.feedback_act[index]=self.w[index]*qvsl/((self.w[index]*self.pj[index]-qvsl)*3.6)
                else:
                    qout[index]=min(self.vf[index]*p[index],self.w_[index]*(self.pj_[index]-p[index]),self.Cd[index])
                    qin0=qout[index]-u[index]*(p[index]-self.pc[index])
                    qvsl=np.median((0,qin0,self.vf[index]*self.pc[index]))
                    self.feedback_act[index]=self.w[index]*qvsl/((self.w[index]*self.pj[index]-qvsl)*3.6)

        #self.feedback_act=np.clip(1.1*self.feedback_act,8.33,25.00)
        self.feedback_act[0] = np.median([8.33, rl_act[0], 22.22])
        self.feedback_act[1] = np.median([8.33, rl_act[1], 22.22])
        self.feedback_act[2] = np.median([8.33, rl_act[2], 22.22])





    #####################  a new round simulation  #################### 
    def reset(self):

        self.simulation_step = 0
        self.warmup_time=600
        #self.writenewtrips()
        sumoCmd = [self.sumoBinary, "-c", sumoConfig, "--start"]
        traci.start(sumoCmd)
        
        state_overall=0
        self.density=[[] for _ in range(len(self.discharge))]#用于统计和计算密度的列表
        self.density_R=[]

        self.last_act = np.array([22.22, 22.22, 22.22])#np.array([27.78, 27.78, 27.78])
        self.feedback_act=np.array([22.22, 22.22, 22.22]) #np.array([25.00, 25.00, 25.00])
        #self.R =50
        arrived_veh=0
        while self.simulation_step<self.warmup_time:
            traci.simulationStep()
            self.simulation_step += 1
            arrived_veh=arrived_veh+traci.simulation.getArrivedNumber()
            if self.simulation_step>=self.warmup_time-self.control_horizon:
                state_overall = state_overall + self.get_step_state()/self.control_horizon
                for index,lane_id in enumerate(self.discharge):
                    self.density[index].append(traci.lane.getLastStepVehicleNumber(lane_id)*1000/traci.lane.getLength(lane_id))
                R_vehnum=0
                R_length=0
                for lane_id in self.Ramp_list:
                    R_vehnum=R_vehnum+traci.lane.getLastStepVehicleNumber(lane_id)
                    R_length=R_length+traci.lane.getLength(lane_id)
                self.density_R.append(R_vehnum*1000/R_length)


        #self.Feedback_ACT([100,100,100,])
        state_overall=np.concatenate((state_overall,self.feedback_act/22.22,self.greentime))


        return state_overall

    #####################  Set random seed  ####################
    def seed(self, seed):
        self.seedvalue=seed

    #####################  run one step: reward is outflow  #################### 
    def step(self, rl_act):
        state_overall = 0
        reward = 0
        co = 0
        hc = 0
        nox = 0
        pmx = 0
        totalspent=0
        waitingtime=0
        out_flow = 0
        speed=[]
        vehtotal=0
        state_out=[]
        bspeed = 0
        tspeed=0
        random.seed(self.seedvalue)#设置随机种子

        Tspeed=[]
        emergency=0
        haltnumber=0
        flag=False
        done=False
        arrived_veh=0
        #强化学习对feedback控制进行修改
        FB_para=rl_act
       # RM_para=rl_act[-1]

        greentime=self.greentime
        real_act=np.clip(self.feedback_act,8.33,27.78)
        real_act = np.round(real_act / 1.3889) * 1.3889
        for index in range(len(self.discharge)-1):
            if abs(real_act[index+1]-real_act[index])>8.333:
                real_act[index+1]=real_act[index]+np.clip(real_act[index+1]-real_act[index],-8.333,8.333)
        #print(true_act)
        for index in range(len(self.discharge)):
            if abs(real_act[index] - self.last_act[index]) > 5.556:
                real_act[index] = self.last_act[index] + np.clip(real_act[index] - self.last_act[index], -5.556, 5.556)
        self.set_vsl(real_act)
        true_act = np.append(real_act, greentime)
        for i in range(self.control_horizon):
            """
            t=i%30
            t_r=t
            if t==0:
                for i in range(2):
                    if real_act[i]>real_act[i+1] and traci.lane.getLastStepVehicleNumber(self.VSLlist[i+1])>0:
                        prob=0.05*(real_act[i]-real_act[i+1])#比例法计算换道概率
                        cnum=min(int(traci.lane.getLastStepVehicleNumber(self.VSLlist[i+1])*prob),8)
                        v=traci.lane.getLastStepVehicleIDs(self.VSLlist[i+1])
                        if cnum>0 and len(v)>0 :
                            vidlist=random.sample(v,min(cnum,len(v)))
                            #print(cnum)
                            #print(vidlist)
                            for vid in vidlist:
                                traci.vehicle.changeLane(vid,i,10)
                    elif real_act[i]<=real_act[i+1] and traci.lane.getLastStepVehicleNumber(self.VSLlist[i])>0:
                        prob = 0.05*(real_act[i+1] - real_act[i])  # 比例法计算换道概率
                        cnum = min(int(traci.lane.getLastStepVehicleNumber(self.VSLlist[i]) * prob),8)
                        v = traci.lane.getLastStepVehicleIDs(self.VSLlist[i])
                        #print(cnum)
                        if cnum>0 and len(v)>0:
                            vidlist = random.sample(v, min(cnum,len(v)))
                            #print(vidlist)
                            for vid in vidlist:
                                traci.vehicle.changeLane(vid, i+1, 10)
            
            if traci.lanearea.getJamLengthMeters(self.que_detector)>=500 and t==0:
                flag=True
                #reward=reward-6.0
            elif traci.lanearea.getJamLengthMeters(self.que_detector)<500 and t==0:
                flag=False
            if t<600 or flag:#全绿灯
                traci.trafficlight.setPhase('ONR', 0)
            else:
                traci.trafficlight.setPhase('ONR', 1)
            
            if (traci.lanearea.getJamLengthMeters(self.que_detector[0]) >= 400 or traci.lanearea.getJamLengthMeters(
                    self.que_detector[1]) >= 400) and t_r == 0:
                flag = True
                # reward=reward-6.0
            elif t_r == 0:
                flag = False
            if t_r < greentime[0] or flag:
                traci.trafficlight.setPhase('J0', 0)
            else:
                traci.trafficlight.setPhase('J0', 1)
            """
            state_overall = state_overall + self.get_step_state()/self.control_horizon
            #arrived_veh=arrived_veh+traci.simulation.getArrivedNumber()
            
            #反馈控制相关的状态计算
            for index,lane_id in enumerate(self.discharge):
                self.density[index].append(traci.lane.getLastStepVehicleNumber(lane_id)*1000/traci.lane.getLength(lane_id))
            R_vehnum=0
            R_length=0
            for lane_id in self.Ramp_list:
                R_vehnum=R_vehnum+traci.lane.getLastStepVehicleNumber(lane_id)
                R_length=R_length+traci.lane.getLength(lane_id)
            self.density_R.append(R_vehnum*1000/R_length)
            #print('Ramp',R_length,R_vehnum,R_vehnum*1000/R_length)
            #out_flow = out_flow + self.calc_outflow()
            #waitingtime=waitingtime+self.calc_waitingtime()
            #bspeed = bspeed + self.calc_bottlespeed()
            '''for detector in self.bottleneck_detector:
                dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
                if dspeed >= 0:
                    speed.append(dspeed)'''

            for detector in self.outID:
                veh_list = traci.inductionloop.getLastStepVehicleIDs(detector)
                #print(detector,veh_list)
                if veh_list:
                     state_out.extend(veh_list)
            """
            for detector in self.total_detector:
                tspeed= traci.edge.getLastStepMeanSpeed(detector)
                if tspeed>0 and traci.edge.getLastStepVehicleNumber(detector)>0:
                    Tspeed.append(tspeed*traci.edge.getLastStepVehicleNumber(detector))
                    vehtotal = vehtotal + traci.edge.getLastStepVehicleNumber(detector)
            """
            '''for detector in self.total_detector:
                if traci.edge.getLastStepHaltingNumber(detector)>0:
                    waitingtime=waitingtime+traci.edge.getLastStepHaltingNumber(detector)'''
            #计算通行时间
            
            #totalspent=totalspent+len(traci.simulation.getPendingVehicles())+traci.vehicle.getIDCount()
            traci.simulationStep()
            self.simulation_step += 1
            #emergency=emergency+self.calc_emergency()
             # the reward is defined as the outflow 
            #co_temp, hc_temp, nox_temp, pmx_temp = self.calc_emission()
            #co = co + co_temp/1000 # g
            #hc = hc + hc_temp/1000 # g
            #nox = nox + nox_temp/1000 #g
            #pmx = pmx + pmx_temp/1000
        #if bspeed / self.control_horizon>20:
            #reward=reward+100/ ((emergency + 1))
        #else:
            #reward=reward+bspeed/self.control_horizon/20+out_flow/50

        '''if speed:
            bspeed=np.mean(speed)
        else:
            bspeed=29.06'''
        #out_flow=arrived_veh

        if state_out:
            out_flow=len(set(state_out))
        else:
            out_flow=0
       # print(state_out)
        """
        if Tspeed and vehtotal>0:
            tspeed=np.sum(Tspeed)/vehtotal
        else:
            tspeed=29.06
        """
        #reward=reward+bspeed/10-haltnumber/35+out_flow/60
        #speedvar=0
        #for i in range(len(real_act)):
            #speedvar=speedvar+(real_act[i]*3.6-self.last_act[i]*3.6)**2
        for index in range(len(real_act)):
            self.last_act[index] = real_act[index]
        #reward = reward+out_flow*6/self.control_horizon-speedvar/(len(real_act)*10000)
        reward=reward+out_flow/5.0
        self.Feedback_ACT(FB_para)
        #self.Feedback_ACT(FB_para)
        if self.simulation_step>=4799:
            self.close()
            done=True
        
        #reward=-totalspent/3600-speedvar/(len(real_act)*8000)
        #reward = reward+tspeed/15+out_flow*4/self.control_horizon
        
        #print(bspeed/5,haltnumber/8,waitingtime/4000)
        state_overall = np.concatenate((state_overall, self.feedback_act / 27.78, self.greentime))#state_overall = np.concatenate((state_overall, self.feedback_act / 27.78, self.last_act / 27.78))

        #return state_overall/self.control_horizon, reward, self.simulation_step, out_flow, bspeed/self.control_horizon,co, hc, nox, pmx,emergency,waitingtime
        #state_overall=np.concatenate((state_overall,self.mpc_act/29.06,self.last_act/29.06))
        return state_overall, reward,done,true_act
    
    def close(self):
        traci.close()
        

    
    
    