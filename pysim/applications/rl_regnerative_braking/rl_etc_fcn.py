# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:34:49 2019

@author: Kyunghan
"""
import numpy as np
def fcn_set_vehicle_param(body_model, vehicle_model, parameter_set, parameter_est):
    body_model.Drivetrain_config(conf_rd_wheel = parameter_set['Conf_wheel_rad'][0,0],                                  
                                  conf_jw_mot = parameter_set['Conf_iner_mot'][0,0], 
                                  conf_gear = parameter_set['Conf_gear'][0,0],
                                  conf_mass_veh = parameter_set['Conf_mass_veh_empty'][0,0],
                                  conf_mass_add = parameter_est['ConfEst_add_mass'][0,0],
                                  conf_jw_wheel = parameter_est['ConfEst_iner_wheel'][0,0],
                                  conf_jw_trns_out = parameter_est['ConfEst_iner_shaft'][0,0],
                                  conf_jw_diff_in = 0,conf_jw_diff_out = 0, conf_jw_trns_in = 0)
    body_model.conf_eff_eq_pos = parameter_est['ConfEst_eff_shaft'][0,0]
    body_model.conf_eff_eq_neg = (2-1/body_model.conf_eff_eq_pos)
    vehicle_model.Veh_config(conf_drag_air_coef = parameter_est['ConfEst_drag_air'][0,0],
                            conf_drag_ca = parameter_est['ConfEst_drag_rol_a'][0,0],
                            conf_drag_cc = parameter_est['ConfEst_drag_rol_b'][0,0])    
pass

def fcn_plot_lrn_result(logging_data, rl_data, ax, fig_num):    

    data_rl = logging_data[0].DataProfile
    data_drv = logging_data[1].DataProfile
    data_mod = logging_data[2].DataProfile
    data_ctl = logging_data[3].DataProfile
    reward_norm_dis = rl_data[0]
    reward_sum = rl_data[1]
    prob_model_out = rl_data[2]
    policy_actionprob = rl_data[3]
    
    ax[0].clear(); 
    ax[0].plot(data_rl['action_index'],alpha = 0.7); 
    ax[0].set_title('action')
    ax[6].clear();
    ax[6].plot(data_ctl['trq_reg'], lw = 2, color = 'black',alpha = 0.7, label = 'trq set filt')
    ax[6].plot(data_ctl['trq_reg_raw'],alpha = 0.3, label = 'trq set raw')
    ax[6].legend()
    
    ax[3].clear(); 
    ax[3].plot(data_drv['acc'], lw = 2, color = 'black',alpha = 0.7, label = 'driving')
    ax[3].plot(data_mod['acc'], alpha = 0.3, label = 'model'); 
    ax[3].plot(data_ctl['acc'], alpha = 0.3, label = 'control'); 
    ax[3].legend()
    ax[3].set_title('acc')
    
    ax[5].clear(); 
    ax[5].plot(data_drv['vel'], lw = 2, color = 'black',alpha = 0.7, label = 'driving')
    ax[5].plot(data_mod['vel'], alpha = 0.3, label = 'model'); 
    ax[5].plot(data_ctl['vel'], alpha = 0.3, label = 'control'); 
    ax[5].set_title('vel')
    
    ax[1].clear(); ax[1].plot(policy_actionprob,alpha = 0.7); ax[1].set_title('action prob')
    ax[4].clear(); ax[4].plot(reward_norm_dis,alpha = 0.7); ax[4].set_title('reward array')
    ax[7].clear(); ax[7].plot(data_rl['rv_sum'], alpha = 0.7, label = 'sum')
    ax[7].plot(data_rl['rv_drv'], alpha = 0.7, label = 'driving')
    ax[7].plot(data_rl['rv_mod'], alpha = 0.7, label = 'model')
    ax[7].plot(data_rl['rv_saf'], alpha = 0.7, label = 'safety')
    ax[7].set_title('reward')
    ax[7].legend()

    ax[2].clear(); ax[2].imshow(prob_model_out, cmap = 'Blues', aspect = 'auto'); ax[2].set_title('log(action prob)')    
    ax[8].scatter(fig_num, reward_sum, s = 2, alpha = 0.7)
    
    plt.pause(0.05)
    
pass

class MovAvgFilt:
    def __init__(self, filt_num):
        self.filt_bump = np.zeros(filt_num)
    
    def filt(self, current_data):
        self.filt_bump[0:-1] = self.filt_bump[1:]
        self.filt_bump[-1] = current_data
        filt_data = np.mean(self.filt_bump)
        return filt_data