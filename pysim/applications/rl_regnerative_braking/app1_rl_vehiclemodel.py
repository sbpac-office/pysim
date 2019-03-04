# -*- coding: utf-8 -*-
"""
Application: Vehicle modeling validation in car-following condition
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
* [18/05/31] - Initial release - kyunghan
"""
#%% 0. import modules
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
import pickle
import time
# Set initial path
base_dir = os.path.abspath('..')
env_dir = os.path.abspath('..\data_roadxy')
conf_dir = os.path.abspath('..\data_config')
test_dir = os.path.abspath('..\sim_test')
data_dir = os.path.abspath('..\data_vehmodel\kona_ev_midan')
print('Base directory: ', base_dir)
sys.path.append(base_dir);
sys.path.append(env_dir);
sys.path.append(conf_dir);
sys.path.append(test_dir);
sys.path.append(data_dir);
# Import package lib modules
from model_powertrain import Mod_Power
from model_vehicle import Mod_Body, Mod_Veh
from model_maneuver import Mod_Behavior, Mod_Driver
from model_environment import Mod_Env
from sub_type_def import type_DataLog, type_drvstate
from sub_utilities import Filt_LowPass
from data_roadxy import get_roadxy
from data_config import get_config
#%% 1. Set vehicle parameters    
# Load data set
get_config.set_dir(data_dir)
kona_param = get_config.load_mat('CoeffSet_Kona.mat')
kona_param_est = get_config.load_mat('CoeffOpt_Kona.mat')['CoeffSet_Init']

# Powertrain import and configuration
kona_power = Mod_Power()
# Bodymodel import and configuration
kona_drivetrain = Mod_Body()
# Set parameters
kona_drivetrain.Drivetrain_config(conf_rd_wheel = kona_param['Conf_wheel_rad'][0,0],                                  
                                  conf_jw_mot = kona_param['Conf_iner_mot'][0,0], 
                                  conf_gear = kona_param['Conf_gear'][0,0],
                                  conf_mass_veh = kona_param['Conf_mass_veh_empty'][0,0],
                                  conf_mass_add = kona_param_est[0,3],
                                  conf_jw_wheel = kona_param_est[0,2],
                                  conf_jw_trns_out = kona_param_est[0,0],
                                  conf_jw_diff_in = 0,conf_jw_diff_out = 0, conf_jw_trns_in = 0)

kona_drivetrain.conf_eff_eq_pos = kona_param_est[0,1]
kona_drivetrain.conf_eff_eq_neg = (2-1/kona_drivetrain.conf_eff_eq_pos)
# Vehicle set
kona_vehicle = Mod_Veh(kona_power, kona_drivetrain)
kona_vehicle.Veh_config(conf_drag_air_coef = kona_param_est[0,4],
                        conf_drag_ca = kona_param_est[0,5],
                        conf_drag_cc = kona_param_est[0,6])
# Driver model
drv_kyunghan = Mod_Driver()
drv_kyunghan.I_gain_lat = 0.001
drv_kyunghan.P_gain_lat = 0.002
drv_kyunghan.I_gain_yaw = 0.3
drv_kyunghan.P_gain_yaw = 0.5

# Behavior model
beh_driving = Mod_Behavior(drv_kyunghan)
#%% 2. Simulation config
#  2. brake torque
data_driver = get_config.load_mat('Data_Kyunghan.mat')
data_result = get_config.load_mat('SimResult_Kyunghan.mat')
u_trq_mot_set = data_driver['Trq_Mot']
u_trq_brk_set = data_driver['Trq_BrkMec']
sim_time_range = data_driver['Veh_Time'] 
result_veh_vel = data_driver['Veh_Vel']

u_trq_drag = data_result['simresult_trq_drag']
Ts = 0.01         
#%% 3. Simulation
# Set logging data
sim_vehicle = type_DataLog(['veh_vel','veh_acc','wdot_veh'])
sim_states = type_DataLog(['w_mot','w_shaft','w_wheel','trq_shaft_in','trq_wheel_load','trq_drag'])
sim_wheel = type_DataLog(['t_wheel_in','t_wheel_traction_f'])
veh_vel = result_veh_vel[0]
w_wheel = data_driver['Rot_Wheel'][0]
w_mot = w_wheel*kona_drivetrain.conf_gear
w_shaft = w_wheel
kona_drivetrain.w_vehicle = w_wheel

start_time = time.time()
for sim_step in range(len(sim_time_range)):
    # Arrange cruise input
    u_t_mot = u_trq_mot_set[sim_step]
    u_t_brk = u_trq_brk_set[sim_step]
    u_t_reg = 0
    # Vehicle model sim       
    #   1. Drag force        
    t_drag, f_drag = kona_vehicle.Drag_system(veh_vel)
#    t_drag = u_trq_drag[sim_step]
#    f_drag = u_trq_drag[sim_step]/kona_drivetrain.conf_rd_wheel;
    #   2. Torque equivalence
    t_mot_load, t_shaft_in, t_shaft_out, t_wheel_in, t_wheel_traction_f, t_driven, f_lon = kona_vehicle.ModDrive.Lon_equivalence(u_t_mot,u_t_brk,t_drag)
    #   3. Dynamics of component
    w_mot = kona_vehicle.ModDrive.Motor_dynamics(u_t_mot, t_mot_load, w_mot)
    w_shaft = kona_vehicle.ModDrive.Driveshaft_dynamics(t_shaft_in, t_shaft_out, w_shaft)
    w_wheel = kona_vehicle.ModDrive.Tire_dynamics(t_wheel_in, t_wheel_traction_f, u_t_brk/4, w_wheel)
    #   4. Vehicle driven
    veh_vel, veh_acc = kona_vehicle.Veh_lon_dynamics(f_lon, f_drag, veh_vel)
#        veh_vel = kona_vehicle.Veh_lon_driven(u_acc_in, u_brk_in)
    
    sim_vehicle.StoreData([veh_vel,veh_acc,kona_drivetrain.w_dot_vehicle])
    sim_states.StoreData([w_mot,w_shaft,w_wheel,t_shaft_in,t_wheel_in,t_drag])
    sim_wheel.StoreData([t_wheel_in,t_wheel_traction_f])
end_time = time.time()
print('Exc time: ',end_time - start_time)
for name_var in sim_vehicle.NameSet:
    globals()['sim_'+name_var] = sim_vehicle.get_profile_value_one(name_var)
for name_var in sim_states.NameSet:
    globals()['sim_'+name_var] = sim_states.get_profile_value_one(name_var)
for name_var in sim_wheel.NameSet:
    globals()['sim_'+name_var] = sim_wheel.get_profile_value_one(name_var)
#%%
plt.plot(sim_t_wheel_traction_f,label='trac')    
plt.plot(sim_t_wheel_in,label='in')    
plt.legend()

#%%
plt.figure()
plt.plot(sim_time_range, result_veh_vel*3.6,label='vehicle data')
plt.plot(sim_time_range, sim_veh_vel*3.6, label='py_sim')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [km/h]')
plt.legend()