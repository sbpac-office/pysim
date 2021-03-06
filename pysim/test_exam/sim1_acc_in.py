# -*- coding: utf-8 -*-
"""
Test code: acc in test
==============================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Test scenario
~~~~~~~~~~~~~
* Set acc and brake position

Test result
~~~~~~~~~~~~~
* Vehicle longitudinal velocity
* Input values

Update
~~~~~~~~~~~~~
* [18/05/31] - Initial release - kyunghan
"""
if __name__== "__main__":
    #%% 0. Import python lib modules    
    import matplotlib.pyplot as plt
    import numpy as np
    import os    
    # set initial path
    base_dir = os.path.abspath('..\..')
    print('Base directory: ', base_dir)
    os.chdir(base_dir)    
    from pysim.models.model_vehicle import Mod_Veh, Mod_Body
    from pysim.models.model_power import Mod_Power
##    from models.model_maneuver import Mod_Behavior, Mod_Driver
##    from models.model_environment import Mod_Env
    from pysim.sub_util.sub_type_def import type_DataLog
    #%% 1. Import models
    # Powertrain import and configuration
    kona_power = Mod_Power()
    #%%
    # ~~~~~
    # Bodymodel import and configuration
    kona_body = Mod_Body()    
    # ~~~~
    # Vehicle set
    kona_vehicle = Mod_Veh(kona_power, kona_body)

    #%% 2. Simulation config
    Ts = 0.01
    sim_time = 40
    sim_time_range = np.arange(0,sim_time,0.01)

    # ----------------------------- select input set ---------------------------
    Input_index = 2
    if Input_index == 0:
    # Go straight : Input_index = 0
        u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.9))))
        u_brk_val = 0 * np.ones(len(sim_time_range))
    elif Input_index == 1:
    # Sin wave : Input_index = 1
        u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 + 0.1*np.sin(0.01*np.arange(int(len(sim_time_range)*0.9)))))
        u_brk_val = 0 * np.ones(len(sim_time_range))
    # Brake : Input_index = 2
    elif Input_index == 2:
        u_acc_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.1)), 0.3 * np.ones(int(len(sim_time_range)*0.4)),  0 * np.ones(int(len(sim_time_range)*0.5))))
        u_brk_val = np.concatenate((0 * np.ones(int(len(sim_time_range)*0.55)), 0.3 * np.ones(int(len(sim_time_range)*0.45))))
    else:
        print('입력을 똑바로 하세요 ~_~')

    #%% 3. Run simulation
    # Set logging data
    sim1 = type_DataLog(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Brk_Set','SOC'])
    w_veh_deb = []
    for sim_step in range(len(sim_time_range)):
        # Arrange vehicle input
        u_acc_in = u_acc_val[sim_step]
        u_brk_in = u_brk_val[sim_step]
        # Vehicle model sim
        [veh_vel, the_wheel] = kona_vehicle.Veh_driven(u_acc = u_acc_in, u_brake = u_brk_in)
        [pos_x, pos_y, pos_s, pos_n, psi_veh] = kona_vehicle.Veh_position_update(veh_vel, the_wheel)
        SOC = kona_vehicle.ModPower.ModBattery.SOC
        # Store data
        sim1.StoreData([veh_vel, pos_x, pos_y, u_acc_in, u_brk_in, SOC])
        w_veh_deb.append(kona_vehicle.ModDrive.w_vehicle)

    [sim1_veh_vel, sim1_pos_x, sim1_pos_y, sim1_u_acc, sim1_u_brk, sim1_soc] = sim1.get_profile_value(['Veh_Vel','Pos_X','Pos_Y','Acc_Set','Brk_Set','SOC'])
    #%% 4. Result plot
    fig = plt.figure(figsize=(8,4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(224)
    ax1.plot(sim1_pos_x, sim1_pos_y);ax1.set_xlabel('X position [m]');ax1.set_ylabel('Y position [m]')
    ax2.plot(sim_time_range, sim1_veh_vel)
    ax3.plot(sim_time_range, sim1_u_acc,label='Acc')
    ax3.plot(sim_time_range, sim1_u_brk,label='Brk')
    ax3.legend()
