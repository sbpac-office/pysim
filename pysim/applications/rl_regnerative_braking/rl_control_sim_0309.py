# -*- coding: utf-8 -*-
"""
Application: Smart regenerative braking based on reinforment learning
======================================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Design smart regenerative braking module using RL algorithm
* Reflect driver characteristics in braking situation

Update
~~~~~~~~~~~~~
* [19/02/13] - Initial draft design
* [19/03/03] - Reinforcement learning algorithm 
* [19/03/05] - Agent change: ddrqn (double deep reccurent q learning)
"""
# import python libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import scipy.io as sio
import copy
import random
import pickle
import scipy.linalg
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, LSTM, Reshape
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Sequential, load_model

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) 
app_dir = os.path.abspath('')
os.chdir('..\..\..')
from pysim.models.model_power import Mod_Power, Mod_Battery, Mod_Motor
from pysim.models.model_vehicle import Mod_Body, Mod_Veh
from pysim.models.model_environment import Mod_Env
from pysim.models.model_maneuver import Mod_Driver, Mod_Behavior
from pysim.sub_util.sub_type_def import type_DataLog, type_pid_controller
from pysim.sub_util.sub_utilities import Filt_LowPass
os.chdir(app_dir)
# import application modules
import get_data
get_data.set_dir(os.path.abspath('.\driving_data'))
DrivingData = get_data.load_mat('CfData1.mat')

get_data.set_dir(os.path.abspath('.\driver_data'))
DriverData = get_data.load_mat('LrnVecCf.mat')

get_data.set_dir(os.path.abspath('.\model_data'))
kona_param = get_data.load_mat('CoeffSet_Kona.mat')
kona_param_est = get_data.load_mat('CoeffEst_Kona.mat')
Ts = 0.01

from rl_idm import IdmAccCf, DecelStateRecog, IdmClassic
from rl_environment import EnvRegen
from rl_algorithm import DdqrnAgent
from rl_etc_fcn import MovAvgFilt, fcn_plot_lrn_result, fcn_set_vehicle_param, fcn_log_data_store
#%%
def fcn_epdata_arrange(ep_data, model, dis_fac):    
    data_length = len(ep_data)
    state_in_size = data_length - model_conf['input_sequence_num']
    input_state = np.zeros((state_in_size, model_conf['input_sequence_num'], model_conf['input_num']))
    q_array = np.zeros((state_in_size, model_conf['action_dim']))
    q_max_array = np.zeros((state_in_size))
    action_index_array = np.zeros((state_in_size))
    reward_array = np.zeros((state_in_size))
    q_from_reward = np.zeros((state_in_size))
    
            
    for i in range(state_in_size):
        ep_data_seq_set = ep_data[i:i+model_conf['input_sequence_num']]
        for j in range(model_conf['input_sequence_num']):
            input_state[i,j,:] = ep_data_seq_set[j][0]
        input_state_dim = np.expand_dims(input_state[i],axis = 0)
        q_array[i,:] = model.predict(input_state_dim)
        q_max_array[i] = np.max(q_array[i,:])
        action_index_array[i] = np.argmax(q_array[i,:])
        reward_array[i] = ep_data[i+model_conf['input_sequence_num']-1][2]
    
    sum_val = 0        
    for step_index in reversed(range(0, state_in_size)):
        sum_val = sum_val * dis_fac + reward_array[step_index]
        q_from_reward[step_index] = sum_val
        
    return input_state, q_array, q_max_array, action_index_array, reward_array, q_from_reward

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain    
    # K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    K_lqr = np.matrix((B.T*X)/R) 
    eigVals, eigVecs = scipy.linalg.eig(A-B*K_lqr)
     
    return K_lqr, X, eigVals


A = np.array(((0,0),(-1,0)))
B = np.array((1,0))
B = np.reshape(B, (2,1))

Q = np.array(((0.1,0),(0,0.001)))
R = np.array((1))

K_lqr,X,eigVals = lqr(A,B,Q,R)

#%% 1. Pysim Model import
# Powertrain model import
kona_power = Mod_Power()
# Body model import
kona_drivetrain = Mod_Body()
# Vehicle model import
kona_vehicle = Mod_Veh(kona_power, kona_drivetrain)
# Driver model import
drv_kyunghan = Mod_Driver()
# Behavior model import
beh_driving = Mod_Behavior(drv_kyunghan)
# Set model parameter
fcn_set_vehicle_param(kona_drivetrain, kona_vehicle, kona_param, kona_param_est)


# RL controller
# Idm
idm_kh = IdmAccCf(DriverData)
cf_state_recog = DecelStateRecog()
idm_cls = IdmClassic()
# Agent
K.clear_session()
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)


model_conf = {'input_num': 8, 'input_sequence_num': 8, 'action_dim': 6, 'lrn_rate': 0.0005}
agent_conf = {'min_data_length': 200}

agent_reg = DdqrnAgent(model_conf['input_num'], model_conf['input_sequence_num'],  model_conf['action_dim'], model_conf['lrn_rate'])
agent_reg.explore_dn_freq = 2000

acc_set_filt = MovAvgFilt(7)
reg_trq_ctl = type_pid_controller()
reg_trq_ctl.P_gain = 50
reg_trq_ctl.I_gain = 500
# Env
env_reg = EnvRegen(kona_power.ModBattery.SOC)


#%% 2. Simulation setting
swt_plot = 'on'
#%% 2. Simulation
sim_vehicle = type_DataLog(['time','veh_vel_measure','veh_vel_pre','veh_vel','veh_acc','acc_measure',
                            'drv_aps_in','drv_bps_in','trq_mot','w_mot','w_shaft','w_wheel','reldis_measure'])

sim_algorithm = type_DataLog(['stDrvInt','stRegCtl','acc_set_lqr','acc_set_classic','x1','x2'])

sim_idm = type_DataLog(['stBrkSection','acc_est','acc_ref','vel_est','vel_ref',
                        'reldis_est','dis_eff','dis_adj','dis_adj_delta',
                        'param_reldis_init','param_reldis_adj','flag_idm_run'])
    
sim_rl = type_DataLog(['state_array','action_index','rv_sum','rv_mod','rv_drv','rv_saf','rv_eng'])
sim_rl_mod = type_DataLog(['acc','vel','reldis','accref','dis_eff','vel_ref','time','section'])
sim_rl_drv = type_DataLog(['acc','reldis','vel','prevel'])
sim_rl_ctl = type_DataLog(['acc','accref','reldis','relvel','vel','trq_reg','soc','acc_set','acc_set_idm','acc_set_lqr'])

# Set figure plot
fig = plt.figure(figsize = (11,9)); ax = []
for i in range(9):
    ax.append(plt.subplot(3,3,i+1))

# Set initial flag and state
flagRegCtlInit = 0
kona_vehicle.swtRegCtl = 2
cf_state_recog.stRegCtl = 'driving'
model_cnt = 0

# Agent configuration
# torque_set = np.array([-1, -0.5, 0, 0.5, 1])
# torque_set = np.array([240, 200, 160, 120, 80, 40, 0])
co_factor_set = np.array([0,0.2,0.4,0.6,0.8,1])

reward_sum_array = []
reward_array = []
episode_num = 0
fig_num = 0



#agent_reg.model.load_weights('factor_oncase.h5')
#agent_reg.target_model.load_weights('factor_oncase.h5')
#%%
for it_num in range(1):
    # for sim_step in range(len(DrivingData['Data_Time'])):
    sim_vehicle.set_reset_log()
    sim_algorithm.set_reset_log()
    sim_idm.set_reset_log()   
    
    regen_on_array = []
    array_action = []
    reward_array = []
    a_lqr = 0
    a_idm = 0
    x1 = 0
    x2 = 0
    for sim_step in range(len(DrivingData['Data_Time'])):
#    for sim_step in range(29000, 31500):
        # Road measured driving data
        sim_time = DrivingData['Data_Time'][sim_step]
        acc_veh_measure_step = DrivingData['DataVeh_Acc'][sim_step]
        vel_veh_measure_step = DrivingData['DataVeh_Vel'][sim_step]
        vel_preveh_measure_step = DrivingData['DataVeh_VelPre'][sim_step]
        driver_aps_in_step = DrivingData['DataDrv_Aps'][sim_step]
        driver_bps_in_step = DrivingData['DataDrv_Brk'][sim_step]
        rel_dis_step = DrivingData['DataRad_RelDis'][sim_step]
        motor_torque_step = DrivingData['DataVeh_MotTorque'][sim_step]
        motor_rotspd_step = DrivingData['DataVeh_MotRotSpeed'][sim_step]
        # Transition state
        stDrvInt = cf_state_recog.pedal_transition_machine(driver_aps_in_step, driver_bps_in_step)
        stRegCtl = cf_state_recog.regen_control_machine(stDrvInt, rel_dis_step)
        
        # Vehicle data for initial lization
        driving_data = {'acc':acc_veh_measure_step, 'vel':vel_veh_measure_step, 'reldis': rel_dis_step, 'prevel': vel_preveh_measure_step}
        pre_vel = vel_preveh_measure_step
        # Prediction sampling time conversion
        if stDrvInt == 'acc off' or stDrvInt == 'acc on':
            model_cnt = 0        
        else:
            model_cnt = model_cnt + 1     
        
        # Prediction each 10ms
        if model_cnt%10 == 0:
            idm_kh.stBrkState = idm_kh.state_def(idm_kh.mod_profile, idm_kh.stBrkState, stDrvInt, driving_data, pre_vel)
            idm_kh.mod_profile = idm_kh.profile_update(idm_kh.stBrkState, idm_kh.mod_profile, pre_vel)
            acc_from_classic, dis_eff = idm_cls.get_acc_set(driving_data)
        
            
        # Control
        if stRegCtl == 'reg on':            
            ''' =================== Regen control algorithm ===================='''
            ''' Regen control using rl_agent policy gradient
                1. state definition
                2. get action - torque index
                3. reward calculation
            '''
            "1. State definition"            
            
            # Initialization
            if flagRegCtlInit == 0:
                rel_dis_pre = rel_dis_step
                control_result = driving_data
                control_result['prevel'] = pre_vel
                control_result['relvel'] = pre_vel - vel_veh_measure_step
                control_result['accref'] = -0.5*(pre_vel**2 - vel_veh_measure_step**2)/rel_dis_step
                flagRegCtlInit = 1
                trqRegSet = 0
                state_in_sqs = np.zeros((1,model_conf['input_sequence_num'],model_conf['input_num']))
                cnt_episode = 0                
                acc_set_filt.filt(acc_veh_measure_step)
                rel_dis_array = []
            else:
                rel_dis_pre = rel_dis
                cnt_episode = cnt_episode + 1
            # ===== Model state update
            model_data = idm_kh.mod_profile
            model_data['time'] = model_cnt*0.01
            model_data['section'] = idm_kh.flag_idm_state
            # print('mod_time: ',model_cnt, 'mod_section: ', model_data['section'])
            # ===== Get state array
            state_in = env_reg.get_norm_state_value(model_data, control_result)
            
            if cnt_episode+1 < model_conf['input_sequence_num']:
                state = 'observe'
                state_in_sqs[0,cnt_episode,:] = state_in
            else:
                state_in_sqs[0,0:-1,:] = state_in_sqs[0,1:,:]
                state_in_sqs[0,-1,:] = state_in
                state = 'control'
                
            "2. Action"
            "Regen control based on rl agent"
            
            ''' Regen from rl '''
            # trq_reg_set_delta = torque_set[action_index]
            # trqRegSet = trq_ctller.filt(trq_reg_set_delta)
            
            "LQR control"
            x1 = -control_result['relvel']
            x2 = control_result['reldis'] - 4*control_result['prevel']
            a_lqr = -(K_lqr[0,0]*x1 + K_lqr[0,1]*x2)
            a_idm = idm_kh.mod_profile['acc']
            
            if state == 'observe':
                action_index = 0          
            else:
                action_index = agent_reg.get_action(state_in_sqs)
            co_factor = co_factor_set[action_index]
            
            acc_set = co_factor*a_lqr + (1-co_factor)*a_idm
            acc_set = acc_set_filt.filt(acc_set)
            
            trqRegSet = reg_trq_ctl.Control(control_result['acc'], acc_set)
            
            ''' Regen from logging data '''
            
            
            # ===== Vehicle driven
            kona_vehicle.t_mot_reg_set = trqRegSet
            drv_aps = 0
            drv_bps = 0
            kona_vehicle.Veh_lon_driven(drv_aps, drv_bps)
            # ===== Vehicle state and relative state update
            veh_vel = kona_vehicle.vel_veh
            rel_vel = pre_vel - veh_vel
            rel_dis = rel_dis_pre + Ts*rel_vel
            veh_acc = kona_vehicle.veh_acc
            acc_ref = 0.5*(pre_vel**2 - vel_veh_measure_step**2)/rel_dis_step
            "3. Reward calculation"
            control_result = {'acc': veh_acc, 'vel': veh_vel, 'relvel': pre_vel - veh_vel, 'reldis': rel_dis_pre, 'prevel': pre_vel, 'accref':acc_ref}
            rv_sum = env_reg.get_reward(driving_data,model_data, control_result, kona_power.ModBattery.SOC)
            reward_array.append(rv_sum)
            "4. Data logging and store sample"
            # ===== Update vehicle state
            state_target = env_reg.get_norm_state_value(model_data, control_result)            
            
            if state == 'control':
                agent_reg.memory.store_sample(state_in, action_index, rv_sum, state_target)
            
            sim_rl.StoreData([state_in, action_index, rv_sum, env_reg.rv_model, env_reg.rv_driving, env_reg.rv_safety, env_reg.rv_energy])    
            
            sim_rl_mod.StoreData([model_data['acc'],model_data['vel'],model_data['reldis'],model_data['acc_ref'],
                                  model_data['dis_eff'],model_data['vel_ref'],model_data['time'],model_data['section']])
            
            sim_rl_drv.StoreData([driving_data['acc'], driving_data['reldis'],driving_data['vel'], pre_vel])
            
            sim_rl_ctl.StoreData([control_result['acc'], control_result['accref'], control_result['reldis'],
                                  control_result['relvel'],control_result['vel'], trqRegSet, kona_power.ModBattery.SOC, acc_set, a_idm, a_lqr])
            
            rel_dis_array.append(rel_dis_step)
            regen_on_array.append(1)
    
        else:        
            flagRegCtlInit = 0
            # Set vehicle state to measurement data
            veh_vel = vel_veh_measure_step
            rel_dis = rel_dis_step
            kona_vehicle.veh_acc = acc_veh_measure_step
            kona_power.t_mot = motor_torque_step
            kona_power.w_mot = motor_rotspd_step
            kona_drivetrain.w_motor = motor_rotspd_step
            kona_drivetrain.w_shaft = motor_rotspd_step/kona_drivetrain.conf_gear
            kona_drivetrain.w_vehicle = vel_veh_measure_step/kona_vehicle.conf_rd_wheel
            kona_drivetrain.w_wheel = motor_rotspd_step/kona_drivetrain.conf_gear        
            drv_aps = 0
            drv_bps = 0 
            
            regen_on_array.append(0)
        
        
        if stDrvInt == 'acc on':
            # Episode done
            "Save episode data"
            ep_data = agent_reg.memory.episode_experience
            q_max, loss = agent_reg.train_from_replay()
            
            if agent_reg.flag_train_state == 1:                
                print('##train on batch, rv_sum:  loss: ',loss, ' lrn_num: ', agent_reg.lrn_num, ' explore: ', agent_reg.epsilon)
                      
            if len(ep_data) < agent_conf['min_data_length']:         
                print('##acc on learning fail: less than min length')
                agent_reg.memory.episode_experience = []
            elif (np.array(rel_dis_array) >= 100).any():
                print('##object loss')
                agent_reg.memory.episode_experience = []
            else:
                agent_reg.memory.add_episode_buffer()                
                if (episode_num-1)%1 == 0:
                        logging_data = [sim_rl, sim_rl_drv, sim_rl_mod, sim_rl_ctl]
                        ep_data_arry = fcn_epdata_arrange(ep_data,  agent_reg.model, agent_reg.dis_fac)
                        fcn_plot_lrn_result(logging_data, ep_data_arry, ax, fig_num)
                        fig_num = fig_num + 1
                    
            episode_num = episode_num+1

            sim_rl.set_reset_log()
            sim_rl_mod.set_reset_log()
            sim_rl_drv.set_reset_log()
            sim_rl_ctl.set_reset_log()
        
    
        
#        reward_array = []
        
        # Learning to episode
               
        # [drv_aps, drv_bps] = beh_driving.Lon_control(vel_veh_measure_step,kona_vehicle.vel_veh)
        # kona_vehicle.Veh_lon_driven(drv_aps, drv_bps)
        # Set vehicle state when     
        kona_vehicle.vel_veh = veh_vel
        
            
        # Data store
        sim_vehicle.StoreData([sim_time, vel_veh_measure_step, vel_preveh_measure_step, kona_vehicle.vel_veh, kona_vehicle.veh_acc, acc_veh_measure_step,
                               driver_aps_in_step, driver_bps_in_step, kona_power.t_mot, kona_power.w_mot, kona_drivetrain.w_shaft, kona_drivetrain.w_wheel, rel_dis])
        sim_algorithm.StoreData([stDrvInt, stRegCtl,a_lqr,acc_from_classic, x1, x2])
        sim_idm.StoreData([idm_kh.stBrkState, idm_kh.mod_profile['acc'], idm_kh.mod_profile['acc_ref'], 
                           idm_kh.mod_profile['vel'], idm_kh.mod_profile['vel_ref'], idm_kh.mod_profile['reldis'], 
                           idm_kh.mod_profile['dis_eff'], idm_kh.param_active['DisAdj'], idm_kh.param_active['DisAdjDelta'],
                           idm_kh.param_active['RelDisInit'], idm_kh.param_active['RelDisInit'], idm_kh.flag_idm_run])        
    "End of driving data iteration"
    reward_sum_array.append(np.sum(reward_array))
    print('!!!!!!!!!!!!!!!! Episode is terminated, Reward_Sum: ', reward_sum_array[-1])
    if reward_sum_array[-1] >= -223200:
                        fcn_log_data_store([logging_data, ep_data_arry,reward_sum_array],'factor_onecase_bestresult_fin.pkl')    
                        agent_reg.model.save_weights("factor_drivingcase_best.h5")
                        print('!!============================== result convergen ==================================!!')
    # Set preceding vehicle
#%% 3. Result plot       
if swt_plot == 'on':    
    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    
    ax1.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_acc'], label='acc')
    ax1.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['acc_est'], label='acc set model')
    # ax1.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['acc_ref'], label='acc ref model')
    ax1.plot(sim_vehicle.DataProfile['time'], sim_algorithm.DataProfile['acc_set_lqr'], label='acc set lqr')
    # ax1.plot(sim_vehicle.DataProfile['time'], sim_algorithm.DataProfile['acc_set_classic'], label='acc set classic')
    ax1.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['acc_measure'], label='acc measure')    
    ax1.legend()
    ax1.set_ylim(-4, 4)

    ax2.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel'], label='vel veh')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel_pre'], label='vel pre')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel_measure'], label='vel measure')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['vel_est'], label='vel est')
    ax2.legend()
    
    # ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['trq_mot'],label = 'mot torq')
    ax3.plot(sim_vehicle.DataProfile['time'], regen_on_array,label = 'rel dis')
    # ax3.plot(sim_vehicle.DataProfile['time'], sim_algorithm.DataProfile['x2'],label = 'x2')
    ax3.legend()
    
    #%% figure for model
    plt.figure(figsize = (5,9))
    ax1 = plt.subplot(611)
    ax2 = plt.subplot(612, sharex=ax1)
    ax3 = plt.subplot(613, sharex=ax1)
    ax4 = plt.subplot(614, sharex=ax1)
    ax5 = plt.subplot(615, sharex=ax1)
    ax6 = plt.subplot(616, sharex=ax1)
    ax1.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['stBrkSection'], label='brake section')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_acc'], label='acc')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['acc_est'], label='acc est')
    ax2.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['acc_measure'], label='acc set')
    # ax2.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['acc_ref'], label='acc ref')
    ax2.legend()
    ax2.set_ylim(-4, 4)
    ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel'], label='vel veh')
    ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel_pre'], label='vel pre')
    ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['veh_vel_measure'], label='vel set')
    ax3.legend()
    ax4.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['reldis_est'], label='reldis est')
    ax4.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['reldis_measure'], label='reldis')
    ax4.set_ylim(0,50)
    ax4.legend()
    ax5.plot(sim_vehicle.DataProfile['time'], regen_on_array, label='reg on swt')
#    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_eff_cls'], label='dis eff classic')    
#    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_adj'], label='dis adj')
#    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_adj_delta'], label='dis adj delta')
    ax5.legend()
    ax6.plot(sim_vehicle.DataProfile['time'], sim_algorithm.DataProfile['stDrvInt'],label = 'drv int')
    #%% figure for reinforcement learning
    plt.figure(figsize = (5,9))
    ax1 = plt.subplot(611); ax2 = plt.subplot(612, sharex=ax1); ax3 = plt.subplot(613, sharex=ax1)
    ax4 = plt.subplot(614, sharex=ax1); ax5 = plt.subplot(615, sharex=ax1); ax6 = plt.subplot(616, sharex=ax1)
    # ax1.plot(np.array(sim_rl.DataProfile['state_array'])[:,0], label='state veh acc')
    ax1.plot(np.array(sim_rl.DataProfile['state_array']), label='state')
    ax2.plot(sim_rl.DataProfile['rv_sum'], label = 'rv sum')
    ax2.plot(sim_rl.DataProfile['rv_mod'], label = 'rv model')
    ax2.plot(sim_rl.DataProfile['rv_drv'], label = 'rv driving')
    ax2.plot(sim_rl.DataProfile['rv_saf'], label = 'rv safety')
    ax2.legend()
    #%%
# K.clear_session()
# get_data.set_dir(os.path.abspath('.\driving_data'))
# DrivingData = get_data.load_mat('CfData1.mat')

sim_time = DrivingData['Data_Time']
acc_veh_measure_step = DrivingData['DataVeh_Acc']
vel_veh_measure_step = DrivingData['DataVeh_Vel']
vel_preveh_measure_step = DrivingData['DataVeh_VelPre']
driver_aps_in_step = DrivingData['DataDrv_Aps']
driver_bps_in_step = DrivingData['DataDrv_Brk']
rel_dis_step = DrivingData['DataRad_RelDis']
motor_torque_step = DrivingData['DataVeh_MotTorque']
motor_rotspd_step = DrivingData['DataVeh_MotRotSpeed']
rel_vel = vel_veh_measure_step - vel_preveh_measure_step

param_am = 0.63
param_b = 1.67
param_vref = 20
param_delta = 4
param_T = 2
param_s0 = 3

eff_dis = param_s0 + param_T*vel_veh_measure_step + vel_veh_measure_step*rel_vel/(2*np.sqrt(param_am*param_b))

acc_set = param_am * (1 - pow((vel_veh_measure_step/param_vref),param_delta) - (eff_dis/rel_dis_step)**2)

#     agent_reg.target_model.predict(state_in_sqs)
plt.figure()
plt.plot(acc_veh_measure_step)
plt.plot(acc_set)
plt.ylim((-4,4))
# plt.plot(sim_idm.DataProfile['acc_est'])
#%%
sim_time = DrivingData['Data_Time']
acc_veh_measure_step = DrivingData['DataVeh_Acc']
vel_veh_measure_step = DrivingData['DataVeh_Vel']
vel_preveh_measure_step = DrivingData['DataVeh_VelPre']
driver_aps_in_step = DrivingData['DataDrv_Aps']
driver_bps_in_step = DrivingData['DataDrv_Brk']
rel_dis_step = DrivingData['DataRad_RelDis']
motor_torque_step = DrivingData['DataVeh_MotTorque']
motor_rotspd_step = DrivingData['DataVeh_MotRotSpeed']
rel_vel = vel_veh_measure_step - vel_preveh_measure_step

veh_vel = vel_veh_measure_step
x1 = vel_veh_measure_step - vel_preveh_measure_step
x2 = rel_dis_step - 3*vel_preveh_measure_step
a_lqr = -(K_lqr[0,0]*x1 + K_lqr[0,1]*x2)

plt.figure()
plt.plot(acc_veh_measure_step)
plt.plot(a_lqr)
