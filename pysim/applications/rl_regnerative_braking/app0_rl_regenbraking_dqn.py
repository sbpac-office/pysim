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
* [18/05/31] - Initial release - kyunghan
"""
#%% 0. Import python lib modules
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from keras.layers import Dense, LSTM, Reshape
from keras.optimizers import Adam, SGD
from keras.models import Sequential, load_model
from keras import backend as K
K.clear_session()
#%% Agent class
class agent_mc_reinforce:
    def __init__(self, input_state_num, output_action_num, model_dim_sqelay, model_dim_dense):
        self.conf_lrn_rate = 0.1
        self.conf_dis_fac = 0.9
        self.output_action_num = output_action_num
        self.policy_model = self.set_policy_model(input_state_num, output_action_num, model_dim_sqelay, model_dim_dense)
        self.policy_train_fcn = self.set_mc_optimizer_fcn(output_action_num)
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        
    def set_policy_model(self, input_state_num, output_action_num, model_dim_sqelay, model_dim_dense):
        # Build deep sequential model
        #  1st layer - LSTM sequential layer
        model = Sequential()
        ModSeq1 = LSTM(model_dim_dense, activation = 'tanh', return_sequences=False, 
                       input_shape = (model_dim_sqelay, input_state_num))
        model.add(ModSeq1)
        # 2nd layer - dense layer, softmax activation
        ModDens1 = Dense(output_action_num, activation = 'softmax', 
                         input_shape = (model_dim_sqelay, model_dim_dense))
        model.add(ModDens1)
        # print model structure
        model.summary()
        return model    
    
    def set_mc_optimizer_fcn(self, output_action_num):
        # Determine optimization function
        #   - Arg: input_state, action and reward 
        #          -> Calculate cross entropy loss function
        
        # Set the action place holder
        action_pseudo = K.placeholder(shape = [None, output_action_num])
        value_normdis = K.placeholder(shape = [None, ])

        # Set the action probability - expectation of model output
        action_prob = K.sum(action_pseudo * self.policy_model.output, axis = 1)

        # Set the cross entropy loss function
        cross_entropy = K.log(action_prob) * value_normdis
        loss = -K.sum(cross_entropy)
        
        # Declare train function with optimizer
        optimizer = Adam(lr = self.conf_lrn_rate)
        updates = optimizer.get_updates(self.policy_model.trainable_weights, [], loss)
        train_fcn = K.function([self.policy_model.input, action_pseudo, value_normdis], [], updates = updates)
        return train_fcn
    
    def get_action(self, state):
        # Predict policy probability
        policy_prob = self.policy_model.predict(state)[0]
        self.policy_prob = policy_prob
        
        # Select action index based on policy probability
        action_index = np.random.choice(self.output_action_num, 1, p=policy_prob)[0]
        return action_index
    
    def store_sample(self, state, action_index, reward):
        self.state_array.append(state[0])
        self.action_array.append(action_index)
        self.reward_array.append(reward)    
    
    def calc_norm_values(self, reward_array):        
        # Calculate accumulated rewards with discount factor
        values_dis = np.zeros_like(reward_array)
        sum_val = 0        
        for step_index in range(0, len(reward_array)):
            sum_val = sum_val * self.conf_dis_fac + reward_array[step_index]
            values_dis[step_index] = sum_val
        
        # Calculate normalized value array
        self.values_dis = np.float32(values_dis)
        values_norm = values_dis - np.mean(values_dis)
        values_norm = values_norm / np.std(values_norm)
        self.values_norm = values_norm        
        return values_norm
    
    def train_model(self):
        # Call train function
        #  - Arg: inpout_state_array, model_output_array(action_index), value array
        #      !! action index will convert to action probability with model.output
        values_norm_array = self.calc_norm_values(self.reward_array)
        self.policy_train_fcn([self.state_array, self.action_array, values_norm_array],[])
        # Reset array
        self.state_array = []
        self.action_array = []
        self.reward_array = []        

#%% Environment class       
class env_brake:
    def __init__(self):
        self.r_reg = 0
        self.r_drv = 0
        self.r_saf = 0
        self.soc_old = 0
        self.set_r_coef()        
    
    def set_r_coef(self, reward_conf_vel = 0.1, reward_val_drv = -10, 
                   reward_conf_loc_across_dis = -3, reward_conf_loc_across_spd = 4,  reward_val_location_across = -200, 
                   reward_conf_loc_behind_dis = 5, reward_conf_loc_behind_spd = 16, reward_val_location_behind = -100):
        self.reward_conf_vel = reward_conf_vel
        self.reward_val_drv = reward_val_drv
        self.reward_val_location_behind = reward_val_location_behind
        self.reward_val_location_across = reward_val_location_across
        self.reward_conf_loc_across_dis = reward_conf_loc_across_dis
        self.reward_conf_loc_across_spd = reward_conf_loc_across_spd        
        self.reward_conf_loc_behind_dis = reward_conf_loc_behind_dis
        self.reward_conf_loc_behind_spd = reward_conf_loc_behind_spd
                
    def get_reward_drv(self):
        self.r_drv = self.rc_drv
        return self.r_drv
    
    def get_reward_saf(self, data_vel, data_dis):                
        behind_cri_dis = data_vel*self.reward_conf_loc_behind_spd + self.reward_conf_loc_behind_dis
        across_cri_dis = data_vel*self.reward_conf_loc_across_spd + self.reward_conf_loc_across_dis               
        if data_dis <= behind_cri_dis:
            self.r_saf = self.reward_val_location_behind
        elif data_dis >= across_cri_dis:
            self.r_saf = self.reward_val_location_across
        else:
            self.r_saf = 0
        return self.r_saf
    
    def get_reward_vel(self, velocity_guide_line, velocity_veh):
        self.r_vel = self.rc_vel*abs(velocity_guide_line - velocity_veh)
        return self.r_vel
        
    def get_reward(self,velocity_guide_line, velocity_veh ,data_dis,):
        r_vel = self.get_reward_vel(velocity_guide_line, velocity_veh)
        r_saf = self.get_reward_saf(velocity_veh, data_dis)
        r_drv = self.get_reward_drv()
        self.r_sum = r_vel + r_saf + r_drv
        return self.r_sum