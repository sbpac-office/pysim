# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:34:49 2019

@author: Kyunghan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import scipy
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

def fcn_plot_lrn_result(logging_data, ep_data_arry, ax, fig_num):    

    data_rl = logging_data[0].DataProfile
    data_drv = logging_data[1].DataProfile
    data_mod = logging_data[2].DataProfile
    data_ctl = logging_data[3].DataProfile
    q_array = ep_data_arry[1]
    q_array_max = ep_data_arry[2]
    action_index_from_q = ep_data_arry[3]
    reward_array = ep_data_arry[4]
    q_from_reward = ep_data_arry[5]
    reward_sum = np.sum(reward_array)
    ax[0].clear(); 
    ax[0].plot(data_rl['action_index'],alpha = 0.7,label = 'from epsilon');
    ax[0].plot(action_index_from_q, alpha = 0.7, label = 'from max q')
    ax[0].legend()
    ax[0].set_title('action')
    ax[6].clear();
    ax[6].plot(data_ctl['trq_reg'], lw = 2, color = 'black',alpha = 0.7, label = 'trq set filt')
    # ax[6].plot(data_ctl['trq_reg_raw'],alpha = 0.3, label = 'trq set raw')
    ax[6].legend()
    
    ax[3].clear(); 
    ax[3].plot(data_drv['acc'], lw = 2, color = 'black',alpha = 0.7, label = 'driving')
    ax[3].plot(data_mod['acc'], alpha = 0.3, label = 'model acc set'); 
    ax[3].plot(data_ctl['acc_set_lqr'], alpha = 0.3, label = 'lqr acc set'); 
    ax[3].plot(data_ctl['acc_set'], alpha = 0.3, label = 'merged acc set'); 
    ax[3].plot(data_ctl['acc'], alpha = 0.3, label = 'control result');
    ax[3].legend()
    ax[3].set_title('acc')
    
    ax[5].clear(); 
    ax[5].plot(data_drv['vel'], lw = 2, color = 'black',alpha = 0.7, label = 'driving')
    ax[5].plot(data_mod['vel'], alpha = 0.3, label = 'model'); 
    ax[5].plot(data_ctl['vel'], alpha = 0.3, label = 'control'); 
    ax[5].set_title('vel')
    
    ax[1].clear(); ax[1].plot(q_array, alpha = 0.7); ax[1].set_title('q array from model'); ax[1].legend()
    ax[4].clear(); ax[4].plot(q_from_reward,alpha = 0.7); ax[4].set_title('q array from reward array')
    ax[1].plot(q_array_max,alpha = 0.7); ax[1].set_title('q array from model')
    ax[7].clear(); ax[7].plot(data_rl['rv_sum'], alpha = 0.7, label = 'sum')
    ax[7].plot(data_rl['rv_drv'], alpha = 0.7, label = 'driving')
    ax[7].plot(data_rl['rv_mod'], alpha = 0.7, label = 'model')
    ax[7].plot(data_rl['rv_saf'], alpha = 0.7, label = 'safety')
    ax[7].plot(data_rl['rv_eng'], alpha = 0.7, label = 'energy')
    ax[7].set_title('reward')
    ax[7].legend()

    ax[2].clear(); ax[2].imshow(q_array, cmap = 'Blues', aspect = 'auto'); ax[2].set_title('log(action prob)')    
    # ax[2].clear()
    # ax[2].plot(data_ctl['x_1'],alpha = 0.7,label = 'x_rel_dis')
    # ax[2].plot(data_ctl['x_2'],alpha = 0.7,label = 'x_rel_vel')
    # ax[2].plot(data_ctl['x_1_r'],alpha = 0.7,label = 'x_rel_dis_des')
    # ax[2].legend()
    ax[8].scatter(fig_num, reward_sum, s = 2, alpha = 0.7)
    
    plt.pause(0.05)
    
pass

class MovAvgFilt:
    
    def __init__(self, filt_num):
        self.filt_bump = np.zeros(filt_num)
        self.filt_num = filt_num
    
    def filt(self, current_data):
        self.filt_bump[0:-1] = self.filt_bump[1:]
        self.filt_bump[-1] = current_data
        filt_data = np.mean(self.filt_bump)
        return filt_data
    
    def reset_filt(self,):
        self.filt_bump = np.zeros(self.filt_num)

def fcn_log_data_store(data, filename):
#    filename = 'one_case_result.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(data, output)
        
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

def fcn_driving_data_arrange(driving_data_raw):
    DrivingData = {}
    list_vars = driving_data_raw.dtype.names
    for i in range(len(list_vars)):
        DrivingData[list_vars[i]] = driving_data_raw[list_vars[i]][0,0]
    DrivingData['DataVeh_VelPre'] = DrivingData['DataVeh_Vel'] + DrivingData['DataRad_RelVel']
    return DrivingData


def fcn_epdata_arrange(ep_data, model, dis_fac, model_conf):    
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