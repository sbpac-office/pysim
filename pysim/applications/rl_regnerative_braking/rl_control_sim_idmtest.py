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
from pysim.models.model_power import Mod_Power
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
DriverDataRaw = get_data.load_mat('Results_Sils_Learning.mat')
DriverDataRaw['BaseMap_InitDis'] = np.transpose(DriverDataRaw['BaseMap_InitDis'])
DriverDataRaw['BaseMap_CoastDis'] = np.transpose(DriverDataRaw['BaseMap_CoastDis'])

DriverDataKh = copy.deepcopy(DriverDataRaw)
DriverDataKh['LrnVec_Param_AccSlopeCf'] =  np.expand_dims(np.transpose(DriverDataKh['MapArry_AccSlopeCf'][0,:]), axis = 1)
DriverDataKh['LrnVec_Param_RelDisAdj'] =  np.expand_dims(np.transpose(DriverDataKh['MapArry_AdjDis'][0,:]), axis = 1)
DriverDataKh['LrnVec_Param_RelDisInit'] =  np.expand_dims(np.transpose(DriverDataKh['MapArry_InitDis'][0,:]), axis = 1)

DriverDataGb = copy.deepcopy(DriverDataRaw)
DriverDataGb['LrnVec_Param_AccSlopeCf'] = np.expand_dims(np.transpose(DriverDataGb['MapArry_AccSlopeCf'][2,:]),axis=1)
DriverDataGb['LrnVec_Param_RelDisAdj'] =  np.expand_dims(np.transpose(DriverDataGb['MapArry_AdjDis'][2,:]), axis = 1)
DriverDataGb['LrnVec_Param_RelDisInit'] =  np.expand_dims(np.transpose(DriverDataGb['MapArry_InitDis'][2,:]), axis = 1)

DriverDataYk = copy.deepcopy(DriverDataRaw)
DriverDataYk['LrnVec_Param_AccSlopeCf'] =  np.expand_dims(np.transpose(DriverDataYk['MapArry_AccSlopeCf'][1,:]), axis = 1)
DriverDataYk['LrnVec_Param_RelDisAdj'] = np.expand_dims(np.transpose(DriverDataYk['MapArry_AdjDis'][1,:]), axis = 1)
DriverDataYk['LrnVec_Param_RelDisInit'] =  np.expand_dims(np.transpose(DriverDataYk['MapArry_InitDis'][1,:]), axis = 1)

get_data.set_dir(os.path.abspath('.\model_data'))
kona_param = get_data.load_mat('CoeffSet_Kona.mat')
kona_param_est = get_data.load_mat('CoeffEst_Kona.mat')
Ts = 0.01

from rl_idm import IdmAccCf, DecelStateRecog, IdmClassic
from rl_environment import EnvRegen
from rl_algorithm import DdqrnAgent
from rl_etc_fcn import MovAvgFilt, fcn_plot_lrn_result, fcn_set_vehicle_param, fcn_log_data_store, fcn_epdata_arrange, lqr
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
# idm_kh = IdmAccCf(DriverDataKh)
cf_state_recog = DecelStateRecog()
idm_cls = IdmClassic()

# LQR controller
A = np.array(((0,0),(-1,0)))
B = np.array((1,0))
B = np.reshape(B, (2,1))

Q = np.array(((0.1,0),(0,0.001)))
R = np.array((1))

K_lqr,X,eigVals = lqr(A,B,Q,R)

# Lower controller
acc_set_filt = MovAvgFilt(21)
reg_trq_ctl = type_pid_controller()
reg_trq_ctl.P_gain = 50
reg_trq_ctl.I_gain = 500
#%% 2. Simulation setting
swt_plot = 'off'
#%% 2. Simulation
sim_vehicle = type_DataLog(['time','veh_vel_measure','veh_vel_pre','veh_vel','veh_acc','acc_measure',
                            'drv_aps_in','drv_bps_in','trq_mot','w_mot','w_shaft','w_wheel','reldis_measure'])

sim_algorithm = type_DataLog(['stDrvInt','stRegCtl','acc_set_lqr','acc_set_classic','x1','x2'])

sim_idm = type_DataLog(['stBrkSection','acc_est','acc_ref','vel_est','vel_ref',
                        'reldis_est','dis_eff','dis_adj','dis_adj_delta',
                        'param_reldis_init','param_reldis_adj','flag_idm_run'])

# Set initial flag and state
flagRegCtlInit = 0
kona_vehicle.swtRegCtl = 2
cf_state_recog.stRegCtl = 'driving'
model_cnt = 0

reward_sum_array = []
reward_array = []
episode_num = 0
fig_num = 0

plt.figure()

idm_logging_data = []
for it_num in range(3):
    if it_num == 0:        
        idm_kh = IdmAccCf(DriverDataKh)
        idm_kh.termvelfac = 2        
    elif it_num == 1:        
        idm_kh = IdmAccCf(DriverDataYk)
        idm_kh.termvelfac = 1.5
    else:
        it_num = DriverDataGb
        idm_kh = IdmAccCf(DriverDataGb)
        idm_kh.termvelfac = 1    
    
    # for sim_step in range(len(DrivingData['Data_Time'])):
    sim_vehicle.set_reset_log()
    sim_algorithm.set_reset_log()
    sim_idm.set_reset_log()   
    
    array_reward = []
    array_action = []
    a_lqr = 0
    a_idm = 0
    x1 = 0
    x2 = 0
#    for sim_step in range(len(DrivingData['Data_Time'])):
    for sim_step in range(16750, 18000):
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
            
            # Initialization
            if flagRegCtlInit == 0:
                rel_dis_pre = rel_dis_step
                control_result = driving_data
                control_result['prevel'] = pre_vel
                control_result['relvel'] = pre_vel - vel_veh_measure_step
                control_result['accref'] = -0.5*(pre_vel**2 - vel_veh_measure_step**2)/rel_dis_step
                flagRegCtlInit = 1
                trqRegSet = 0                
                cnt_episode = 0                
                acc_set_filt.filt(acc_veh_measure_step)
                reward_array = []
            else:
                rel_dis_pre = rel_dis
                cnt_episode = cnt_episode + 1
               
                            
            "LQR control"
            x1 = -control_result['relvel']
            x2 = control_result['reldis'] - 4*control_result['prevel']
            a_lqr = -(K_lqr[0,0]*x1 + K_lqr[0,1]*x2)
            
            "Model based control"
            a_idm = idm_kh.mod_profile['acc']
           
            acc_set = a_lqr
            acc_set = acc_set_filt.filt(acc_set)
            
            trqRegSet = reg_trq_ctl.Control(control_result['acc'], acc_set)

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
            control_result = {'acc': veh_acc, 'vel': veh_vel, 'relvel': pre_vel - veh_vel, 'reldis': rel_dis_pre, 'prevel': pre_vel, 'accref':acc_ref}
    
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
        
        "End of driving data iteration"

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
    
    # print(idm_kh.param_active)
    plt.plot(sim_algorithm.DataProfile['x1'])
    plt.plot(sim_algorithm.DataProfile['x2'])
    idm_logging_data.append([copy.deepcopy(sim_vehicle), copy.deepcopy(sim_algorithm), copy.deepcopy(sim_idm)])

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
    
    reldis_state_x2 = np.array(sim_vehicle.DataProfile['reldis_measure']) - 4 * np.array(sim_vehicle.DataProfile['veh_vel_pre'])
    critic_dis =  4 * np.array(sim_vehicle.DataProfile['veh_vel_pre'])
    # ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['trq_mot'],label = 'mot torq')
    ax3.plot(sim_vehicle.DataProfile['time'], sim_vehicle.DataProfile['reldis_measure'],label = 'rel dis')
    ax3.plot(sim_vehicle.DataProfile['time'], critic_dis, label = 'lqr state x2')
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
    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_eff'], label='dis eff')
    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_eff_cls'], label='dis eff classic')    
    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_adj'], label='dis adj')
    ax5.plot(sim_vehicle.DataProfile['time'], sim_idm.DataProfile['dis_adj_delta'], label='dis adj delta')
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
    #%% Acc set from classic idm
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
    #%% Acc set from lqr algorithm
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
