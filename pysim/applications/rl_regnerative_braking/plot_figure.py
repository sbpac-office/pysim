# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:07:54 2019

@author: Kyunghan
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import os
import lib_sbpac
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS) 
Color = lib_sbpac.get_ColorSet()
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
from rl_idm import IdmAccCf, DecelStateRecog, IdmClassic
from rl_environment import EnvRegen
from rl_algorithm import DdqrnAgent
from rl_etc_fcn import MovAvgFilt, fcn_plot_lrn_result, fcn_set_vehicle_param, fcn_log_data_store


def fcn_subplot_set(figure, grid_space, row_index, col_index, xlabel = None, ylabel = None, grid = 'on', row_size = 1, col_size = 1):
    if (row_size == 1) and (col_size == 1):
        ax = figure.add_subplot(grid_space[row_index, col_index])    
    elif row_size == 1:
        ax = figure.add_subplot(grid_space[row_index, col_index:col_index+col_size])
    elif col_size == 1:
        ax = figure.add_subplot(grid_space[row_index:row_index+row_size, col_index])
    else: 
        ax = figure.add_subplot(grid_space[row_index:row_index+row_size, col_index:col_index+col_size])    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid == 'on':
        ax.grid(color = [0.67843137, 0.66666667, 0.67843137],linestyle = '--',alpha = 0.7)        
    return ax   

get_data.set_dir(os.path.abspath('.\driving_data'))
DrivingData = get_data.load_mat('CfData1.mat')
BatteryData = get_data.load_mat('CfData1_Battery.mat')

get_data.set_dir(os.path.abspath('.\model_data'))
kona_param = get_data.load_mat('CoeffSet_Kona.mat')
kona_param_est = get_data.load_mat('CoeffEst_Kona.mat')

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

u_trq_mot_set = DrivingData['DataVeh_MotTorque']
u_trq_brk_set = DrivingData['DataVeh_TrqBrkMec']
sim_time_range = DrivingData['Data_Time'] 
result_veh_vel = DrivingData['DataVeh_Vel']
result_veh_acc = DrivingData['DataVeh_Acc']
result_mot_rot_speed = DrivingData['DataVeh_MotRotSpeed']

sim_vehicle = type_DataLog(['veh_vel','veh_acc','wdot_veh'])
sim_states = type_DataLog(['w_mot','w_shaft','w_wheel','trq_shaft_in','trq_wheel_load','trq_drag'])
sim_wheel = type_DataLog(['t_wheel_in','t_wheel_traction_f'])


driver_u_acc = DrivingData['DataDrv_Aps']
driver_u_brk = DrivingData['DataDrv_Brk']
#%% Figure 3 Regeneration and SOC
" vehicle speed control case "
sim_vehicle = type_DataLog(['u_acc','veh_vel','veh_acc','trq_mot','soc','trq_brk','p_motor_elec'])
sim_vehicle_reg = type_DataLog(['veh_vel','veh_acc','trq_mot','soc','trq_brk'])

for sim_case in range(2):
    # Powertrain model import
    kona_power = Mod_Power()
    # Body model import
    kona_drivetrain = Mod_Body()
    # Vehicle model import
    kona_vehicle = Mod_Veh(kona_power, kona_drivetrain)
    if sim_case == 0:
        kona_vehicle.swtRegCtl = 0
    else:
        kona_vehicle.swtRegCtl = 1
        
    # for sim_step in  range(len(sim_time_range)):
    for sim_step in  range(10000):
        vehicle_speed_set = result_veh_vel[sim_step]
        [u_acc, u_brk] = beh_driving.Lon_control(vehicle_speed_set, kona_vehicle.vel_veh)        
        u_acc = sorted((0,u_acc,2))[1]
        # u_acc = driver_u_acc[sim_step]/10
        # u_brk = driver_u_brk[sim_step]/10
        
        # print(kona_vehicle.ModPower.ModBattery.SOC)
        kona_vehicle.Veh_lon_driven(u_acc, u_brk)
        if kona_vehicle.swtRegCtl == 0:
            sim_vehicle.StoreData([u_acc, kona_vehicle.vel_veh, kona_vehicle.veh_acc, kona_power.ModMotor.t_mot, kona_vehicle.ModPower.ModBattery.SOC , kona_vehicle.t_brake, kona_power.ModMotor.p_mot_elec])
        else:
            sim_vehicle_reg.StoreData([kona_vehicle.vel_veh, kona_vehicle.veh_acc, kona_power.ModMotor.t_mot, kona_vehicle.ModPower.ModBattery.SOC , kona_vehicle.t_brake])
    

plot_index_start = 0
plot_index_end = 10000 # 100 s    
plot_index = range(plot_index_start,plot_index_end)

plt.figure(figsize = (10,3))
ax1 = plt.subplot(131) #"ax1 - input torque "
ax2 = plt.subplot(132, sharex=ax1) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(133, sharex=ax1) #"ax3 - rotational velocity of moto"
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(sim_time_range[plot_index], sim_vehicle.DataProfile['veh_vel'][plot_index_start:plot_index_end], label = 'model result', lw = 1)
ax1.set_ylabel('Vehicle speed [m/s]')
ax1.set_xlabel('Time [s]')

ax2.plot(sim_time_range[plot_index], sim_vehicle.DataProfile['trq_mot'][plot_index_start:plot_index_end], label = 'without regeneration', lw = 1)
ax2.plot(sim_time_range[plot_index], sim_vehicle_reg.DataProfile['trq_mot'][plot_index_start:plot_index_end], label = 'with regeneration', lw = 1, ls = '--', alpha = 0.8)    
ax2.legend()
ax2.set_ylabel('Motor torque [Nm]')
ax2.set_xlabel('Time [s]')

ax3.plot(sim_time_range[plot_index], sim_vehicle.DataProfile['soc'][plot_index_start:plot_index_end], label = 'without regeneration', lw = 1)
ax3.plot(sim_time_range[plot_index], sim_vehicle_reg.DataProfile['soc'][plot_index_start:plot_index_end], label = 'with regeneration', lw = 1, ls = '--', alpha = 0.8)   
ax3.set_ylabel('SOC [%]')
ax3.set_xlabel('Time [s]')
ax3.legend()

plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig3_regeneration_result.png', format='png', dpi=500)    
    
    

#%% Figure 4 vehicle modeling - parameter identification
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

u_trq_mot_set = DrivingData['DataVeh_MotTorque']
u_trq_brk_set = DrivingData['DataVeh_TrqBrkMec']
sim_time_range = DrivingData['Data_Time'] 
result_veh_vel = DrivingData['DataVeh_Vel']
result_veh_acc = DrivingData['DataVeh_Acc']
result_mot_rot_speed = DrivingData['DataVeh_MotRotSpeed']

sim_vehicle = type_DataLog(['veh_vel','veh_acc','wdot_veh'])
sim_states = type_DataLog(['w_mot','w_shaft','w_wheel','trq_shaft_in','trq_wheel_load','trq_drag'])
sim_wheel = type_DataLog(['t_wheel_in','t_wheel_traction_f'])

veh_vel = result_veh_vel[0]
w_wheel = veh_vel/0.318
w_mot = w_wheel*kona_drivetrain.conf_gear
w_shaft = w_wheel
kona_drivetrain.w_vehicle = w_wheel

for sim_step in  range(len(sim_time_range)):
    u_t_mot = u_trq_mot_set[sim_step]
    u_t_brk = u_trq_brk_set[sim_step]
    
    t_drag, f_drag = kona_vehicle.Drag_system(veh_vel)
    
    t_mot_load, t_shaft_in, t_shaft_out, t_wheel_in, t_wheel_traction_f, t_driven, f_lon = kona_vehicle.ModDrive.Lon_equivalence(u_t_mot,u_t_brk,t_drag)
    
    w_mot = kona_vehicle.ModDrive.Motor_dynamics(u_t_mot, t_mot_load, w_mot)
    w_shaft = kona_vehicle.ModDrive.Driveshaft_dynamics(t_shaft_in, t_shaft_out, w_shaft)
    w_wheel = kona_vehicle.ModDrive.Tire_dynamics(t_wheel_in, t_wheel_traction_f, u_t_brk/4, w_wheel)
    
    veh_vel, veh_acc = kona_vehicle.Veh_lon_dynamics(f_lon, f_drag, veh_vel)
    
    sim_vehicle.StoreData([veh_vel,veh_acc,kona_drivetrain.w_dot_vehicle])
    sim_states.StoreData([w_mot,w_shaft,w_wheel,t_shaft_in,t_wheel_in,t_drag])
    sim_wheel.StoreData([t_wheel_in,t_wheel_traction_f])

plot_index_start = 0
plot_index_end = 10000 # 100 s    
plot_index = range(plot_index_start,plot_index_end)

plt.figure(figsize = (10,3))
ax1 = plt.subplot(141) #"ax1 - input torque "
ax2 = plt.subplot(142, sharex=ax1) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(143, sharex=ax1) #"ax3 - rotational velocity of moto"
ax4 = plt.subplot(144, sharex=ax1) #"ax3 - rotational velocity of moto"
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax4.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(sim_time_range[plot_index], u_trq_mot_set[plot_index], label = 'motor torque', lw = 1)
ax1.plot(sim_time_range[plot_index], u_trq_brk_set[plot_index], label = 'brake torque', lw = 1, ls = '--')    
ax1.set_ylabel('Torque [Nm]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax2.plot(sim_time_range[plot_index], sim_vehicle.DataProfile['veh_acc'][plot_index_start:plot_index_end], label = 'model result', lw = 1)
ax2.plot(sim_time_range[plot_index], result_veh_acc[plot_index], label = 'driving data', lw = 1, ls = '--', alpha = 0.8)    
ax2.set_ylabel('Acceleration [m/s^2]')
ax2.set_xlabel('Time [s]')
ax2.legend()
ax3.plot(sim_time_range[plot_index], sim_states.DataProfile['w_mot'][plot_index_start:plot_index_end], label = 'model result', lw = 1)
ax3.plot(sim_time_range[plot_index], result_mot_rot_speed[plot_index], label = 'driving data', lw = 1, ls = '--', alpha = 0.8)    
ax3.set_ylabel('Shaft rotational speed [rad/s]')
ax3.set_xlabel('Time [s]')
ax3.legend()
ax4.plot(sim_time_range[plot_index], sim_states.DataProfile['w_mot'][plot_index_start:plot_index_end], label = 'model result', lw = 1)
ax4.plot(sim_time_range[plot_index], result_mot_rot_speed[plot_index], label = 'driving data', lw = 1, ls = '--', alpha = 0.8)    
ax4.set_ylabel('Shaft rotational speed [rad/s]')
ax4.set_xlabel('Battery current [A]')
ax4.legend()
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig4_modeling_result.png', format='png', dpi=500)
#%% Figure 7 Model parameter and index
get_data.set_dir(os.path.abspath('.\driver_data'))
DriverParam = get_data.load_mat('DrvChar_Idm.mat')

x_reldis_cst = []
x_reldis_init = []
x_index = []

y_reldis_init = []
y_reldis_adj = []
y_accslope = []

for i in range(3):
    if i == 0:
        dataset = DriverParam['DrvChar_KH']
    elif i == 1:
        dataset = DriverParam['DrvChar_YK']
    else:
        dataset = DriverParam['DrvChar_GB']
    
    x_reldis_cst.append(dataset['CoastDis'][0,0])
    x_reldis_init.append(dataset['InitDis'][0,0])
    x_index.append(dataset['InitIndex'][0,0])        
    
    y_reldis_init.append(dataset['InitDis'][0,0])
    y_reldis_adj.append(dataset['AdjDis'][0,0])
    y_accslope.append(dataset['AccSlopeCf'][0,0])        
    
plt.figure(figsize = (10,3))
ax1 = plt.subplot(131) #"ax1 - input torque "
ax2 = plt.subplot(132) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(133) #"ax3 - rotational velocity of moto"
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.scatter(x_index[0], y_accslope[0], label = 'Driver 1', alpha = 0.5, color = Color['RP'][8])
ax1.scatter(x_index[1], y_accslope[1], label = 'Driver 2', alpha = 0.5, color = Color['BP'][4])
ax1.scatter(x_index[2], y_accslope[2], label = 'Driver 3', alpha = 0.5, color = Color['YP'][3])
ax1.legend()
ax1.set_xlabel('Coast index')
ax1.set_ylabel('Acc slope [m/s^3]')
ax1.set_xlim(4,13)

ax2.scatter(x_reldis_cst[0], y_reldis_init[0], label = 'Driver 1', alpha = 0.5, color = Color['RP'][8])
ax2.scatter(x_reldis_cst[1], y_reldis_init[1], label = 'Driver 2', alpha = 0.5, color = Color['BP'][4])
ax2.scatter(x_reldis_cst[2], y_reldis_init[2], label = 'Driver 3', alpha = 0.5, color = Color['YP'][3])
ax2.legend()
ax2.set_xlabel('Coast rel dis [m]')
ax2.set_ylabel('Init rel dis [m]')

ax3.scatter(x_reldis_init[0], y_reldis_adj[0], label = 'Driver 1', alpha = 0.5, color = Color['RP'][8])
ax3.scatter(x_reldis_init[1], y_reldis_adj[1], label = 'Driver 2', alpha = 0.5, color = Color['BP'][4])
ax3.scatter(x_reldis_init[2], y_reldis_adj[2], label = 'Driver 3', alpha = 0.5, color = Color['YP'][3])
ax3.legend()
ax3.set_xlabel('Init rel dis [m]')
ax3.set_ylabel('Adj rel dis [m]')

plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig7_modelparameters.png', format='png', dpi=500)
#%% Figure 8 Learning results - Model parameter vector
get_data.set_dir(os.path.abspath('.\driver_data'))
LrnResult = get_data.load_mat('Results_sils_Learning.mat')

plt.figure(figsize = (10,3))
ax1 = plt.subplot(131) #"ax1 - input torque "
ax2 = plt.subplot(132) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(133) #"ax3 - rotational velocity of moto"
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.scatter(x_index[0], y_accslope[0], label = 'Parameter of driver 1', alpha = 0.5, color = Color['RP'][8], s = 10)
ax1.scatter(x_index[1], y_accslope[1], label = 'Parameter of driver 2', alpha = 0.5, color = Color['BP'][4], s = 10)
ax1.scatter(x_index[2], y_accslope[2], label = 'Parameter of driver 3', alpha = 0.5, color = Color['YP'][3], s = 10)
p1, = ax1.plot(LrnResult['BaseMap_InitIndexCf'], LrnResult['LrnVec_Param_AccSlopeCf'], color = Color['WP'][4], label = 'Base vector', lw = 3, alpha = 0.5)
p2, = ax1.plot(LrnResult['BaseMap_InitIndexCf'], LrnResult['MapArry_AccSlopeCf'][0,:], color = Color['RP'][3], label = 'Updated vector of driver 1')
p3, = ax1.plot(LrnResult['BaseMap_InitIndexCf'], LrnResult['MapArry_AccSlopeCf'][1,:], color = Color['BP'][4], label = 'Updated vector of driver 2')
p4, = ax1.plot(LrnResult['BaseMap_InitIndexCf'], LrnResult['MapArry_AccSlopeCf'][2,:], color = Color['YP'][3], label = 'Updated vector of driver 3')
ax1.legend((p1,p2,p3,p4), ('Base vector','Driver 1','Driver 2','Driver 3'))
ax1.set_xlabel('Initial index')
ax1.set_ylabel('Acc slope vector [m/s^3]')
ax1.set_xlim(4,13)

ax2.scatter(x_reldis_cst[0], y_reldis_init[0], label = 'Driver 1', alpha = 0.5, color = Color['RP'][8], s = 10)
ax2.scatter(x_reldis_cst[1], y_reldis_init[1], label = 'Driver 2', alpha = 0.5, color = Color['BP'][4], s = 10)
ax2.scatter(x_reldis_cst[2], y_reldis_init[2], label = 'Driver 3', alpha = 0.5, color = Color['YP'][3], s = 10)
p1, = ax2.plot(np.transpose(LrnResult['BaseMap_CoastDis']), LrnResult['LrnVec_Param_RelDisInit'], color = Color['WP'][4], label = 'Base vector', lw = 3, alpha = 0.5)
p2, = ax2.plot(np.transpose(LrnResult['BaseMap_CoastDis']), LrnResult['MapArry_InitDis'][0,:], color = Color['RP'][3], label = 'Updated vector of driver 1')
p3, = ax2.plot(np.transpose(LrnResult['BaseMap_CoastDis']), LrnResult['MapArry_InitDis'][1,:], color = Color['BP'][4], label = 'Updated vector of driver 2')
p4, = ax2.plot(np.transpose(LrnResult['BaseMap_CoastDis']), LrnResult['MapArry_InitDis'][2,:], color = Color['YP'][3], label = 'Updated vector of driver 3')
# ax1.legend((p1,p2,p3,p4), ('Base vector','Driver 1','Driver 2','Driver 3'))
ax2.set_xlabel('Coast rel dis [m]')
ax2.set_ylabel('Init rel dis [m]')

x_vector_index = np.transpose(LrnResult['BaseMap_InitDis'])
ax3.scatter(x_reldis_init[0], y_reldis_adj[0], label = 'Driver 1', alpha = 0.5, color = Color['RP'][8], s = 10)
ax3.scatter(x_reldis_init[1], y_reldis_adj[1], label = 'Driver 2', alpha = 0.5, color = Color['BP'][4], s = 10)
ax3.scatter(x_reldis_init[2], y_reldis_adj[2], label = 'Driver 3', alpha = 0.5, color = Color['YP'][3], s = 10)
p1, = ax3.plot(x_vector_index, LrnResult['LrnVec_Param_RelDisAdj'], color = Color['WP'][4], label = 'Base vector', lw = 3, alpha = 0.5)
p2, = ax3.plot(x_vector_index, LrnResult['MapArry_AdjDis'][0,:], color = Color['RP'][3], label = 'Updated vector of driver 1')
p3, = ax3.plot(x_vector_index, LrnResult['MapArry_AdjDis'][1,:], color = Color['BP'][4], label = 'Updated vector of driver 2')
p4, = ax3.plot(x_vector_index, LrnResult['MapArry_AdjDis'][2,:], color = Color['YP'][3], label = 'Updated vector of driver 3')
# ax1.legend((p1,p2,p3,p4), ('Base vector','Driver 1','Driver 2','Driver 3'))
ax3.set_xlabel('Init rel dis [m]')
ax3.set_ylabel('Adj rel dis [m]')

plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig8_learningresult.png', format='png', dpi=500)
#%% Figrue 9 Predictionresults for each driver
# from rl_control_sim_idmtest import idm_logging_data

vehicle_data = idm_logging_data[0][0].DataProfile
control_data = idm_logging_data[0][1].DataProfile
model_data =  idm_logging_data[0][2].DataProfile

model_run = np.array(model_data['flag_idm_run'])
model_start_point = np.min(np.where(model_run=='on')[0])
model_end_point = np.max(np.where(model_run=='on')[0])
plt.figure(figsize = (10,3))
ax1 = plt.subplot(121) #"ax1 - input torque "
ax2 = plt.subplot(122) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_data['time'], vehicle_data['acc_measure'], alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[0][2].DataProfile['acc_est'][model_start_point:model_end_point], color = Color['RP'][3], label = 'Driver model 1')
ax1.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[1][2].DataProfile['acc_est'][model_start_point:model_end_point], color = Color['BP'][4], label = 'Driver model 2' )
ax1.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[2][2].DataProfile['acc_est'][model_start_point:model_end_point], color = Color['YP'][3], label = 'Driver model 3' )
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax1.set_ylim(-1.7, 2)

ax2.plot(vehicle_data['time'], vehicle_data['veh_vel_pre'], alpha = 0.5, color = Color['WP'][4], lw = 3, label = 'Preceding vehicle')
ax2.plot(vehicle_data['time'], vehicle_data['veh_vel_measure'], alpha = 0.7, color = Color['WP'][1], lw = 3, label = 'Measured vehlocity')
ax2.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[0][2].DataProfile['vel_est'][model_start_point:model_end_point], color = Color['RP'][3], label = 'Driver model 1' )
ax2.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[1][2].DataProfile['vel_est'][model_start_point:model_end_point], color = Color['BP'][4], label = 'Driver model 2' )
ax2.plot(vehicle_data['time'][model_start_point:model_end_point], idm_logging_data[2][2].DataProfile['vel_est'][model_start_point:model_end_point], color = Color['YP'][3], label = 'Driver model 3' )
ax2.set_ylim(9,25)
ax2.set_ylabel('Velocity [m/s]')
ax2.set_xlabel('Time [s]')
ax2.legend()
plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig9_idm_prediction_result.png', format='png', dpi=500)
#%% Figure 10 Cruise control results based on LQR algorithm
# run rl_control_sim_0424_mpc_onecase

"fcn_log_data_store([sim_vehicle, sim_algorithm, sim_idm, sim_rl_ctl, sim_rl_drv, sim_rl_mod], file_name)"
with open('oncase_mpc_np15.pkl', 'rb') as output:
    logging_np15 = pickle.load(output)
    
control_data = logging_np15[3].DataProfile
drv_data = logging_np15[4].DataProfile
vehicle_data = logging_np15[0].DataProfile
algorithm_data = logging_np15[1].DataProfile

acc_vehicle = vehicle_data['acc_measure']
acc_set_mpc = np.array(control_data['acc_set'])
acc_control = np.array(control_data['acc'])

time = drv_data['time']

reldis_state_x2 = np.array(vehicle_data['reldis_measure']) - 4 * np.array(vehicle_data['veh_vel_pre'])

state_x1 = algorithm_data['x1']
state_x2 = algorithm_data['x2']
state_xr = algorithm_data['xr']

plt.figure(figsize = (4,7))
ax1 = plt.subplot(311) #"ax1 - input torque "
ax2 = plt.subplot(312) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(313) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_data['time'], acc_vehicle, alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(time, acc_set_mpc, alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Deceleration profile from MPC')
ax1.plot(time, acc_control, alpha = 0.9, color = Color['SP'][7], lw = 2, label = 'Control result', ls = '--')
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax1.set_ylim(-2, 2.5)

ax2.plot(vehicle_data['time'], state_x1, color = Color['RP'][3], lw = 2, label = 'Staet x1 - relative distance')
ax2.plot(vehicle_data['time'], state_xr, color = Color['YP'][4], lw = 2, label = 'desired relative distance', ls = '--')
ax2.set_ylabel('Distance [m]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(vehicle_data['time'], state_x2, color = Color['BP'][1], lw = 2, label = 'Staet x2 - relative velocity')
ax3.plot(vehicle_data['time'], np.zeros(len(state_x2)), color = Color['BP'][5], lw = 2, label = 'desired relative velocity', ls = '--')
ax3.set_ylabel('Velocity [m/s]')
ax3.set_xlabel('Time [s]')
ax3.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig10_mpc_control_result.png', format='png', dpi=500)

#%% MPC result according to prediction horizon
with open('oncase_mpc_np15.pkl', 'rb') as output:
    logging_np15 = pickle.load(output)

with open('oncase_mpc_np10.pkl', 'rb') as output:
    logging_np10 = pickle.load(output)

with open('oncase_mpc_np20.pkl', 'rb') as output:
    logging_np20 = pickle.load(output)

    

drv_data = logging_np15[4].DataProfile
vehicle_data = logging_np15[0].DataProfile

control_data = [logging_np10[3].DataProfile,logging_np15[3].DataProfile,logging_np20[3].DataProfile]
algorithm_data = [logging_np10[1].DataProfile,logging_np15[1].DataProfile,logging_np20[1].DataProfile]

acc_vehicle = vehicle_data['acc_measure']
time = drv_data['time']

acc_set_mpc = []; acc_control = []; state_x1 = []; state_x2 = []; state_xr = []; state_x1_err = [];
for i in range(3):
    acc_set_mpc.append(np.array(control_data[i]['acc_set']))
    acc_control.append(np.array(control_data[i]['acc']))
    
    state_x1.append( algorithm_data[i]['x1'])
    state_x2.append( algorithm_data[i]['x2'])
    state_xr.append( algorithm_data[i]['xr'])
    state_x1_err.append(np.array(state_xr[i]) - np.array(state_x1[i]))

plt.figure(figsize = (4,7))
ax1 = plt.subplot(311) #"ax1 - input torque "
ax2 = plt.subplot(312) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(313) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_data['time'], acc_vehicle, alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(time, acc_set_mpc[0], alpha = 0.9, color = Color['RP'][1], lw = 2, label = 'Np: 10')
ax1.plot(time, acc_set_mpc[1], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Np: 15', ls = '--')
ax1.plot(time, acc_set_mpc[2], alpha = 0.9, color = Color['RP'][5], lw = 2, label = 'Np: 20', ls = '-.')

ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax1.set_ylim(-2, 2.5)

ax2.plot(vehicle_data['time'], state_x1_err[0], color = Color['GP'][0], lw = 2, label = 'Np: 10')
ax2.plot(vehicle_data['time'], state_x1_err[1], color = Color['GP'][3], lw = 2, label = 'Np: 15', ls = '--')
ax2.plot(vehicle_data['time'], state_x1_err[2], color = Color['GP'][9], lw = 2, label = 'Np: 20', ls = '-.')
ax2.set_ylabel('State 1 error [m]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(vehicle_data['time'], state_x2[0], color = Color['BP'][1], lw = 2, label = 'Np: 10')
ax3.plot(vehicle_data['time'], state_x2[1], color = Color['BP'][3], lw = 2, label = 'Np: 15', ls = '--')
ax3.plot(vehicle_data['time'], state_x2[2], color = Color['BP'][5], lw = 2, label = 'Np: 20', ls = '--')
ax3.set_ylabel('State 2 error [m/s]')
ax3.set_xlabel('Time [s]')
ax3.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('figExt_mpc_predicHorizon.png', format='png', dpi=500)

#%% MPC result according to q1 gain
with open('oncase_mpc_np15_1.pkl', 'rb') as output:
    logging_q1 = pickle.load(output)

with open('oncase_mpc_np15.pkl', 'rb') as output:
    logging_q4 = pickle.load(output)

with open('oncase_mpc_np15_8.pkl', 'rb') as output:
    logging_q8 = pickle.load(output)

    

drv_data = logging_q4[4].DataProfile
vehicle_data = logging_q4[0].DataProfile

control_data = [logging_q1[3].DataProfile,logging_q4[3].DataProfile,logging_q8[3].DataProfile]
algorithm_data = [logging_q1[1].DataProfile,logging_q4[1].DataProfile,logging_q8[1].DataProfile]

acc_vehicle = vehicle_data['acc_measure']
time = drv_data['time']

acc_set_mpc = []; acc_control = []; state_x1 = []; state_x2 = []; state_xr = []; state_x1_err = [];
for i in range(3):
    acc_set_mpc.append(np.array(control_data[i]['acc_set']))
    acc_control.append(np.array(control_data[i]['acc']))
    
    state_x1.append( algorithm_data[i]['x1'])
    state_x2.append( algorithm_data[i]['x2'])
    state_xr.append( algorithm_data[i]['xr'])
    state_x1_err.append(np.array(state_xr[i]) - np.array(state_x1[i]))

plt.figure(figsize = (4,7))
ax1 = plt.subplot(311) #"ax1 - input torque "
ax2 = plt.subplot(312) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(313) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_data['time'], acc_vehicle, alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(time, acc_set_mpc[0], alpha = 0.9, color = Color['RP'][1], lw = 2, label = 'Q1: 1')
ax1.plot(time, acc_set_mpc[1], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Q1: 4', ls = '--')
ax1.plot(time, acc_set_mpc[2], alpha = 0.9, color = Color['RP'][5], lw = 2, label = 'Q1: 8', ls = '-.')

ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax1.set_ylim(-2, 2.5)

ax2.plot(vehicle_data['time'], state_x1_err[0], color = Color['GP'][0], lw = 2, label = 'Q1: 1')
ax2.plot(vehicle_data['time'], state_x1_err[1], color = Color['GP'][3], lw = 2, label = 'Q1: 4', ls = '--')
ax2.plot(vehicle_data['time'], state_x1_err[2], color = Color['GP'][9], lw = 2, label = 'Q1: 8', ls = '-.')
ax2.set_ylabel('State 1 error [m]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(vehicle_data['time'], state_x2[0], color = Color['BP'][1], lw = 2, label = 'Q1: 1')
ax3.plot(vehicle_data['time'], state_x2[1], color = Color['BP'][3], lw = 2, label = 'Q1: 4', ls = '--')
ax3.plot(vehicle_data['time'], state_x2[2], color = Color['BP'][5], lw = 2, label = 'Q1: 8', ls = '--')
ax3.set_ylabel('State 2 error [m/s]')
ax3.set_xlabel('Time [s]')
ax3.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('figExt_mpc_q1.png', format='png', dpi=500)

#%% MPC result according to q2 gain
with open('oncase_mpc_np15_q2_0.pkl', 'rb') as output:
    logging_q1 = pickle.load(output)

with open('oncase_mpc_np15.pkl', 'rb') as output:
    logging_q4 = pickle.load(output)

with open('oncase_mpc_np15_q2_5.pkl', 'rb') as output:
    logging_q8 = pickle.load(output)

    

drv_data = logging_q4[4].DataProfile
vehicle_data = logging_q4[0].DataProfile

control_data = [logging_q1[3].DataProfile,logging_q4[3].DataProfile,logging_q8[3].DataProfile]
algorithm_data = [logging_q1[1].DataProfile,logging_q4[1].DataProfile,logging_q8[1].DataProfile]

acc_vehicle = vehicle_data['acc_measure']
time = drv_data['time']

acc_set_mpc = []; acc_control = []; state_x1 = []; state_x2 = []; state_xr = []; state_x1_err = [];
for i in range(3):
    acc_set_mpc.append(np.array(control_data[i]['acc_set']))
    acc_control.append(np.array(control_data[i]['acc']))
    
    state_x1.append( algorithm_data[i]['x1'])
    state_x2.append( algorithm_data[i]['x2'])
    state_xr.append( algorithm_data[i]['xr'])
    state_x1_err.append(np.array(state_xr[i]) - np.array(state_x1[i]))

plt.figure(figsize = (4,7))
ax1 = plt.subplot(311) #"ax1 - input torque "
ax2 = plt.subplot(312) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(313) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_data['time'], acc_vehicle, alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(time, acc_set_mpc[0], alpha = 0.9, color = Color['RP'][1], lw = 2, label = 'Q2: 0.02')
ax1.plot(time, acc_set_mpc[1], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Q2: 0.1', ls = '--')
ax1.plot(time, acc_set_mpc[2], alpha = 0.9, color = Color['RP'][5], lw = 2, label = 'Q2: 0.5', ls = '-.')

ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend()
ax1.set_ylim(-2, 3)

ax2.plot(vehicle_data['time'], state_x1_err[0], color = Color['GP'][0], lw = 2, label = 'Q2: 0.02')
ax2.plot(vehicle_data['time'], state_x1_err[1], color = Color['GP'][3], lw = 2, label = 'Q2: 0.1', ls = '--')
ax2.plot(vehicle_data['time'], state_x1_err[2], color = Color['GP'][9], lw = 2, label = 'Q2: 0.5', ls = '-.')
ax2.set_ylabel('State 1 error [m]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(vehicle_data['time'], state_x2[0], color = Color['BP'][1], lw = 2, label = 'Q2: 0.02')
ax3.plot(vehicle_data['time'], state_x2[1], color = Color['BP'][3], lw = 2, label = 'Q2: 0.1', ls = '--')
ax3.plot(vehicle_data['time'], state_x2[2], color = Color['BP'][5], lw = 2, label = 'Q2: 0.5', ls = '--')
ax3.set_ylabel('State 2 error [m/s]')
ax3.set_xlabel('Time [s]')
ax3.legend()
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('figExt_mpc_q2.png', format='png', dpi=500)

#%%  Figure 13 Onecase result
with open('factor_onecase_bestresult_mpc_rsum.pkl','rb') as f:
    onecase_result = pickle.load(f)

reward_sum_array = onecase_result[2]
reward_sum_array_filt = np.zeros(len(reward_sum_array))
reward_array_filter = MovAvgFilt(101)
for i in range(len(reward_sum_array)):
    reward_sum_array_filt[i] = reward_array_filter.filt(reward_sum_array[i])
    
with open('factor_onecase_bestresult_mpc.pkl','rb') as f:
    onecase_result = pickle.load(f)

rl_result_data_set = onecase_result[0][0].DataProfile
action_index_with_explore = rl_result_data_set['action_index']
rl_driver_data_set = onecase_result[0][1].DataProfile
rl_model_data_set = onecase_result[0][2].DataProfile
rl_control_data_set = onecase_result[0][3].DataProfile

time = rl_model_data_set['time']

ep_data_arry = onecase_result[1]
q_array = ep_data_arry[1]
q_array_max = ep_data_arry[2]
action_index_from_q = ep_data_arry[3]
q_from_reward = ep_data_arry[5]
    

'''
sim_rl = type_DataLog(['state_array','action_index','rv_sum','rv_mod','rv_drv','rv_saf','rv_eng'])
sim_rl_mod = type_DataLog(['acc','vel','reldis','accref','dis_eff','vel_ref','time','section'])
sim_rl_drv = type_DataLog(['acc','reldis','vel','prevel'])
sim_rl_ctl = type_DataLog(['acc','accref','reldis','relvel','vel','trq_reg','soc','acc_set','acc_set_idm','acc_set_lqr'])
'''

'''
" Plotting setting "
ax[0:1,0:1] = Acc set, ax[0,1] = trqset, ax[0,2] = SOC
                     , ax[1,1] = relvel, ax[1,2] = reldis
ax[2,0] = Action     , ax[2,1] = reward, ax[2,2] = lrnresult
'''

plt.close('all')
fig_oncase = plt.figure(constrained_layout=True, figsize = (10,6))
gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig_oncase)

ax_acc = fcn_subplot_set(fig_oncase, gs, 0, 0, 'Time [s]', 'Acceleration [m/s^2]', row_size = 2)
ax_trqset = fcn_subplot_set(fig_oncase, gs, 0, 1, 'Time [s]', 'Torque [Nm]')
ax_soc = fcn_subplot_set(fig_oncase, gs, 0, 2, 'Time [s]', 'SOC [%]')
ax_vel = fcn_subplot_set(fig_oncase, gs, 1, 1, 'Time [s]', 'Velocity [m/s]')
ax_dis = fcn_subplot_set(fig_oncase, gs, 1, 2, 'Time [s]', 'Relative distance [m]')
ax_action = fcn_subplot_set(fig_oncase, gs, 2, 0, 'Time [s]', 'Torque [Nm]')
ax_reward = fcn_subplot_set(fig_oncase, gs, 2, 1, 'Time [s]', 'Reward [-]')
ax_lrnresult = fcn_subplot_set(fig_oncase, gs, 2, 2, 'Time [s]', 'Reward_Sum [-]')
#%%
ax_acc.plot(rl_model_data_set['time'], rl_driver_data_set['acc'], label = 'driving data', alpha = 0.7, color = Color['WP'][4], lw = 3,)
ax_acc.plot(rl_model_data_set['time'], rl_control_data_set['acc_set_idm'], label = 'model acc set', alpha = 0.7, color = Color['BP'][4], lw = 2,)
ax_acc.plot(rl_model_data_set['time'], rl_model_data_set['acc'], label = 'model acc set', alpha = 0.7, color = Color['RP'][4], lw = 2, ls = '--')
# ax_acc.plot(rl_model_data_set['time'], rl_control_data_set['acc_set_lqr'], label = 'cruise acc set', alpha = 0.7, color = Color['YP'][3], lw = 2,)
# ax_acc.plot(rl_model_data_set['time'], rl_control_data_set['acc'], label = 'control result', alpha = 0.9, color = Color['RP'][8], lw = 2,)



reward_sum_array_filt[0:101] = reward_sum_array[0:101]

plt.figure(figsize = (10,6))
ax1 = plt.subplot(231);ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2 = plt.subplot(232);ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3 = plt.subplot(233);ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax4 = plt.subplot(234);ax4.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax5 = plt.subplot(235);ax5.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax6 = plt.subplot(236);ax6.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(rl_model_data_set['time'], rl_driver_data_set['acc'], label = 'driving data', alpha = 0.7, color = Color['WP'][4], lw = 3,)
ax1.plot(rl_model_data_set['time'], rl_control_data_set['acc_set_idm'], label = 'model acc set', alpha = 0.7, color = Color['BP'][4], lw = 2,)
ax1.plot(rl_model_data_set['time'], rl_control_data_set['acc_set_lqr'], label = 'cruise acc set', alpha = 0.7, color = Color['YP'][3], lw = 2,)
ax1.plot(rl_model_data_set['time'], rl_control_data_set['acc'], label = 'control result', alpha = 0.9, color = Color['RP'][8], lw = 2,)
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.set_ylim(-2.0, 1.5)
ax1.legend()

ax2.plot(rl_model_data_set['time'], rl_control_data_set['trq_reg'], label = 'Regnerative torque', alpha = 0.9, color = Color['WP'][4], lw = 2,)
ax2.set_ylabel('Torque [Nm]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(rl_model_data_set['time'], rl_driver_data_set['vel'], label = 'driving data', alpha = 0.7, color = Color['WP'][4], lw = 3)
ax3.plot(rl_model_data_set['time'], rl_driver_data_set['prevel'], label = 'preceding vehicle', alpha = 1, color = Color['BP'][4], lw = 2)
ax3.plot(rl_model_data_set['time'], rl_control_data_set['vel'], label = 'control result', alpha = 1, color = Color['RP'][8], lw = 2, ls = '--')
ax3.set_ylabel('Velocity [m/s^2]')
ax3.set_xlabel('Time [s]')
ax3.legend()

ax4.plot(time, action_index_with_explore, label = 'action index with explore', alpha = 0.7, color = Color['WP'][4], lw = 1)
ax4.plot(time[15:], action_index_from_q, label = 'action index from q model', alpha = 1, color = Color['BP'][2], lw = 3)
ax4.legend()
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Action index [-]')
ax4.set_ylim(-1, 8)

ax5.plot(time[15:], q_array_max, label = 'q max from model', alpha = 0.5, color = Color['WP'][4], lw = 2)
ax5.plot(time[15:], q_from_reward, label = 'measured action value', alpha = 1, color = Color['YP'][2], lw = 2)
ax5.legend()
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Action value [-]')
ax5.set_ylim(-80, 20)

ax6.plot(reward_sum_array, label = 'variation', alpha = 0.3, color = Color['RP'][7], lw = 1)
ax6.plot(reward_sum_array_filt, label = 'filted', alpha = 1, color = Color['RP'][8], lw = 2)
ax6.legend()
ax6.set_xlabel('Iteration [-]')
ax6.set_ylabel('Reward sum [-]')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95, hspace=0.35, wspace=0.38)
plt.savefig('fig13_onecase_learning_result.png', format='png', dpi=500)
#%% Figure 14 Driving case 
'''
    sim_vehicle = type_DataLog(['time','veh_vel_measure','veh_vel_pre','veh_vel','veh_acc','acc_measure',
                                'drv_aps_in','drv_bps_in','trq_mot','w_mot','w_shaft','w_wheel','reldis_measure'])
    
    sim_algorithm = type_DataLog(['stDrvInt','stRegCtl','acc_set_lqr','acc_set_classic','x1','x2'])
    
    sim_idm = type_DataLog(['stBrkSection','acc_est','acc_ref','vel_est','vel_ref',
                            'reldis_est','dis_eff','dis_adj','dis_adj_delta',
                            'param_reldis_init','param_reldis_adj','flag_idm_run'])
    
ax1 - Driving data, acceleration, control results in decel condition
ax2 - vehicle velocity and preceding vechiel
ax3 - Model fitted case
ax4 - Cruise control case
'''

with open('driving_case_driver_0_bestcase.pkl','rb') as f:
    driving_result_driver1 = pickle.load(f)

#%%    
vehicle_data = driving_result_driver1[0].DataProfile
vehicle_time = np.array(vehicle_data['time'])
algorithm_data = driving_result_driver1[1].DataProfile
idm_data = driving_result_driver1[2].DataProfile

driving_acc = np.array(vehicle_data['acc_measure'])
model_acc = np.array(idm_data['acc_est'])
cruise_acc = np.array(algorithm_data['acc_set_lqr'])
control_acc = np.array(vehicle_data['veh_acc'])
regen_on = np.array(algorithm_data['stRegCtl'])
regen_on_index = regen_on == 'reg on'
control_acc[~regen_on_index] = 0

plt.figure(figsize = (10,8))
ax1 = plt.subplot(311);ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2 = plt.subplot(312);ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3 = plt.subplot(325);ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax4 = plt.subplot(326);ax4.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_time, vehicle_data['acc_measure'], alpha = 0.3, color = Color['WP'][4], lw = 2, label = 'driving data')
ax1.plot(vehicle_time, idm_data['acc_est'], alpha = 0.5, color = Color['BP'][4], lw = 1, label = 'model acc set', ls = '-.')
ax1.plot(vehicle_time, algorithm_data['acc_set_lqr'], alpha = 0.5, color = Color['YP'][3], lw = 1, label = 'cruise acc set', ls = '--')
ax1.plot(vehicle_time, control_acc, alpha = 0.5, color = Color['RP'][8], lw = 1, label = 'control results')
ax1.legend()
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')

ax2.plot(vehicle_time, regen_on, label = 'regen control condition')
ax2.set_xlabel('Time [s]')
ax2.legend()

model_case_stindex = 17080  
model_case_endindex = 18000
ax3.plot(vehicle_time[model_case_stindex:model_case_endindex], driving_acc[model_case_stindex:model_case_endindex], alpha = 0.3, color = Color['WP'][4], lw = 2, label = 'driving data')
ax3.plot(vehicle_time[model_case_stindex:model_case_endindex], model_acc[model_case_stindex:model_case_endindex], alpha = 0.5, color = Color['BP'][4], lw = 2, label = 'model acc set', ls = '-.')
ax3.plot(vehicle_time[model_case_stindex:model_case_endindex], cruise_acc[model_case_stindex:model_case_endindex], alpha = 0.5, color = Color['YP'][3], lw = 2, label = 'cruise acc set', ls = '--')
ax3.plot(vehicle_time[model_case_stindex:model_case_endindex], control_acc[model_case_stindex:model_case_endindex], alpha = 0.7, color = Color['RP'][8], lw = 2, label = 'control results')
ax3.set_ylabel('Acceleration [m/s^2]')
ax3.set_xlabel('Time [s]')

cruise_case_st_index = 45713
cruise_case_sd_index = 47000
ax4.plot(vehicle_time[cruise_case_st_index:cruise_case_sd_index], driving_acc[cruise_case_st_index:cruise_case_sd_index], alpha = 0.3, color = Color['WP'][4], lw = 2, label = 'driving data')
ax4.plot(vehicle_time[cruise_case_st_index:cruise_case_sd_index], model_acc[cruise_case_st_index:cruise_case_sd_index], alpha = 0.5, color = Color['BP'][4], lw = 2, label = 'model acc set', ls = '-.')
ax4.plot(vehicle_time[cruise_case_st_index:cruise_case_sd_index], cruise_acc[cruise_case_st_index:cruise_case_sd_index], alpha = 0.5, color = Color['YP'][3], lw = 2, label = 'cruise acc set', ls = '--')
ax4.plot(vehicle_time[cruise_case_st_index:cruise_case_sd_index], control_acc[cruise_case_st_index:cruise_case_sd_index], alpha = 0.7, color = Color['RP'][8], lw = 2, label = 'control results')
ax4.set_ylabel('Acceleration [m/s^2]')
ax4.set_xlabel('Time [s]')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.08, right=0.95, hspace=0.35, wspace=0.3)
plt.savefig('fig14_driving_learning_result.png', format='png', dpi=500)

#%% Figure 15 Control results for individual driver
with open('driving_case_driver_0_figure.pkl','rb') as f:
    driving_result_driver1 = pickle.load(f)
    
with open('driving_case_driver_1_figure.pkl','rb') as f:
    driving_result_driver2 = pickle.load(f)
    
with open('driving_case_driver_2_figure.pkl','rb') as f:
    driving_result_driver3 = pickle.load(f)    

#%%    
vehicle_data1 = driving_result_driver1[0].DataProfile
vehicle_data2 = driving_result_driver2[0].DataProfile
vehicle_data3 = driving_result_driver3[0].DataProfile

q_array1 = np.array(driving_result_driver1[4])
q_array1 = np.reshape(q_array1, (61304,6))
max_q_array1 = np.max(q_array1, axis = 1)
action_index1 = np.argmax(q_array1, axis = 1)

q_array2 = np.array(driving_result_driver2[4])
q_array2 = np.reshape(q_array2, (61304,6))
max_q_array2 = np.max(q_array2, axis = 1)
action_index2 = np.argmax(q_array2, axis = 1)

q_array3 = np.array(driving_result_driver3[4])
q_array3 = np.reshape(q_array3, (61304,6))
max_q_array3 = np.max(q_array3, axis = 1)
action_index3 = np.argmax(q_array3, axis = 1)


vehicle_time = np.array(vehicle_data['time'])
algorithm_data = driving_result_driver1[1].DataProfile
idm_data = driving_result_driver1[2].DataProfile
driving_acc = np.array(vehicle_data['acc_measure'])

control_result1 = driving_result_driver1[1].DataProfile
control_result2 = driving_result_driver2[1].DataProfile
control_result3 = driving_result_driver3[1].DataProfile

control_acc_drv1 = np.array(vehicle_data1['veh_acc'])
control_acc_drv2 = np.array(vehicle_data2['veh_acc'])
control_acc_drv3 = np.array(vehicle_data3['veh_acc'])


model_case_stindex = 17080  
model_case_endindex = 17750

plt.figure(figsize = (10,3))
ax1 = plt.subplot(131) #"ax1 - input torque "
ax2 = plt.subplot(132) #"ax2 - acceleration of vehicle "
ax3 = plt.subplot(133) #"ax2 - acceleration of vehicle "
ax1.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax2.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)
ax3.grid(color = Color['WP'][8],linestyle = '--',alpha = 0.7)

ax1.plot(vehicle_time[model_case_stindex: model_case_endindex], driving_acc[model_case_stindex: model_case_endindex], alpha = 0.7, color = Color['WP'][4], lw = 3, label = 'Measured acceleration')
ax1.plot(vehicle_time[model_case_stindex: model_case_endindex], control_acc_drv1[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Driver 1')
ax1.plot(vehicle_time[model_case_stindex: model_case_endindex], control_acc_drv2[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['BP'][4], lw = 2, label = 'Driver 2')
ax1.plot(vehicle_time[model_case_stindex: model_case_endindex], control_acc_drv3[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['YP'][3], lw = 2, label = 'Driver 3')
ax1.set_ylabel('Acceleration [m/s^2]')
ax1.set_xlabel('Time [s]')
ax1.legend(loc = 9)
ax1.set_ylim(-1.7, 1.5)

ax2.plot(vehicle_time[model_case_stindex: model_case_endindex], action_index1[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Driver 1')
ax2.plot(vehicle_time[model_case_stindex: model_case_endindex], action_index2[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['BP'][4], lw = 2, label = 'Driver 2')
ax2.plot(vehicle_time[model_case_stindex: model_case_endindex], action_index3[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['YP'][3], lw = 2, label = 'Driver 3')
ax2.set_ylabel('Aciton index [-]')
ax2.set_xlabel('Time [s]')
ax2.legend()

ax3.plot(vehicle_time[model_case_stindex: model_case_endindex], max_q_array1[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['RP'][3], lw = 2, label = 'Driver 1')
ax3.plot(vehicle_time[model_case_stindex: model_case_endindex], max_q_array2[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['BP'][4], lw = 2, label = 'Driver 2')
ax3.plot(vehicle_time[model_case_stindex: model_case_endindex], max_q_array3[model_case_stindex: model_case_endindex], alpha = 0.9, color = Color['YP'][3], lw = 2, label = 'Driver 3')
ax3.set_ylabel('Maximum Q value [-]')
ax3.set_xlabel('Time [s]')
ax3.legend()

plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.95, hspace=0.3, wspace=0.4)
plt.savefig('fig15_control_result_driver.png', format='png', dpi=500)

