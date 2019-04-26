# -*- coding: utf-8 -*-
"""
Application: Smart regenerative braking based on reinforment learning
======================================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Agent - Regenerative torque controller

Update
~~~~~~~~~~~~~
* [19/02/22] - Initial draft design
"""
#%% 0. Import python lib modules
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random
from random import choice

from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector, Masking
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, Activation, Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adam, rmsprop, Nadam
from keras import backend as K  
import tensorflow as tf

K.clear_session()
#%% Actor critic algorithm
class agent_actor_critic:
    def __init__(self, sess, action_num, action_dim, state_dim, model_actor_hidden_1, model_actor_hidden_2, conf_lrn_rate_actor,
                 model_critic_hidden_1, model_critic_hidden_2, conf_lrn_rate_critic):
        self.sess = sess
        self.conf_lrn_rate_actor = conf_lrn_rate_actor
        self.conf_lrn_rate_critic = conf_lrn_rate_critic
        self.conf_dis_fac = 0.97
        self.value_dis_min = -40
        self.value_dis_max = 2.5
        self.action_num = action_num
        self.actor_model = self.set_actor_model(state_dim, action_num, model_actor_hidden_1, model_actor_hidden_2)
        self.critic_model = self.set_critic_model(state_dim, action_dim, model_critic_hidden_1, model_critic_hidden_2)
        self.set_actor_optimizer(action_num, conf_lrn_rate_actor)
        self.set_critic_optimizer(conf_lrn_rate_critic)
        self.init_data_array()
        
        K.set_session(sess)
        
    def init_data_array(self):
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        self.action_prob_array = []
        self.action_index_array = []
    
    def set_actor_model(self, state_dim, action_num, model_actor_hidden_1, model_actor_hidden_2):
        # build deep neural network model
        model = Sequential()
        ModSeq1 = Dense(model_actor_hidden_1, activation = 'relu',  input_shape = [state_dim])
        model.add(ModSeq1)        
        ModSeq2 = Dense(model_actor_hidden_2, activation = 'relu')
        model.add(ModSeq2)        
        ModSeq3 = Dense(90, activation = 'relu')
        model.add(ModSeq3)        
        ModDens1 = Dense(action_num, activation = 'softmax')
        model.add(ModDens1)
        # print model structure
        model.summary()
        return model
        
    def set_critic_model(self, state_dim, action_dim, model_critic_hidden_1, model_critic_hidden_2): 
        # build deep neural network model
        model = Sequential()
        ModSeq1 = Dense(model_critic_hidden_1, activation = 'relu',  input_shape = [state_dim])
        model.add(ModSeq1)
        ModSeq2 = Dense(model_critic_hidden_2, activation = 'relu')
        model.add(ModSeq2)
        ModSeq3 = Dense(90, activation = 'relu')
        model.add(ModSeq3) 
        ModDens1 = Dense(action_dim, activation = 'linear')
        model.add(ModDens1)
        # print model structure
        model.summary()
        return model
    
    def set_critic_optimizer(self, conf_lrn_rate_critic):
        optimizer = Adam(lr=conf_lrn_rate_critic)
        self.critic_model.compile(loss = 'mse',  optimizer = optimizer)
        
    def set_actor_optimizer(self, action_num, conf_lrn_rate_actor):
        self.action_index_opt = tf.placeholder("float", [None, action_num])
        self.value_critic_opt = tf.placeholder("float", [None, ])
        self.policy_actionprob_tf = tf.reduce_sum(self.action_index_opt*self.actor_model.output,1)
        self.policy_log_tf = tf.log(self.policy_actionprob_tf)
        self.policy_obj_tf = tf.multiply(self.policy_log_tf, self.value_critic_opt)
        cross_entropy = -K.sum(self.policy_obj_tf)
        gradient_tf = tf.gradients(cross_entropy, self.actor_model.trainable_weights)
        grad = zip(gradient_tf, self.actor_model.trainable_weights)
        self.run_opt = tf.train.AdamOptimizer(conf_lrn_rate_actor).apply_gradients(grad)        
        self.sess.run(tf.global_variables_initializer())
    
    def train_actor_model(self, state_array, action_array_index, reward_array):
#        action_array_val_norm = (action_array - 0)/(6-0)
#        self.critic_model_input = np.concatenate((state_array, action_array_val_norm), axis = 1)
        self.critic_model_out = np.reshape(self.critic_model.predict(state_array),-1)
        value_norm, value_dis = self.calc_discounted_values(reward_array)
        values_dis_norm = self.critic_model_out - np.mean(self.critic_model_out)
        values_dis_norm = values_dis_norm/np.std(values_dis_norm)
        values_dis_norm = np.reshape(values_dis_norm, -1)
#        advantage = value_dis - self.critic_model_out               
#        value_dis_norm = self.calc_discounted_values(reward_array)
        self.policy_actionprob_value, self.policy_log_value, self.policy_obj_value, train_result_actor = self.sess.run(
                [self.policy_actionprob_tf, self.policy_log_tf, self.policy_obj_tf, self.run_opt],
                feed_dict = {
                        self.actor_model.input: state_array,
                        self.action_index_opt: action_array_index,
                        self.value_critic_opt: values_dis_norm})
        return train_result_actor
    
    def calc_discounted_values(self, reward_array):
        # Calculate accumulated rewards with discount factor
        values_dis = np.zeros_like(reward_array)
        sum_val = 0        
        for step_index in reversed(range(0, len(reward_array))):
            sum_val = sum_val * self.conf_dis_fac + reward_array[step_index]
            values_dis[step_index] = sum_val
        values_dis = np.float32(values_dis)
        values_norm = values_dis - np.mean(values_dis)
        values_norm = values_norm / (np.std(values_norm)+0.000001)
#        values_norm = (values_dis)/10
        self.values_norm = np.reshape(values_norm, -1)
        self.values_dis = np.reshape(values_dis, -1)
#        self.value_norm = (values_dis - self.value_dis_min)/(self.value_dis_max - self.value_dis_min)*2 - 1
        return self.values_norm, self.values_dis
    
    def train_critic_model(self, state_array, reward_array):
        values_norm, value_dis = self.calc_discounted_values(reward_array)
#        action_array_val_norm = (action_array - 0)/(6-0)
#        critic_model_input = np.concatenate((state_array, action_array_val_norm), axis = 1)        
        train_result_critic = self.critic_model.fit(state_array, value_dis)
        return train_result_critic
        
    def get_action(self, state):
        self.model_out = self.actor_model.predict(state)[0]            
        action_index = np.random.choice(self.action_num, 1, p=self.model_out)[0]
        return action_index
    
    def store_sample(self, state, action_index, action_index_array, action_prob, reward_step):
        self.state_array.append(state[0])
        self.action_array.append(action_index)
        self.action_index_array.append(action_index_array)
        self.reward_array.append(reward_step)
        self.action_prob_array.append(action_prob)
    
        
#%% Policy gradient - MC reinforcement
class AgentMcReinforce:
    def __init__(self, sess, input_state_num, output_action_num, model_dim_dens1, model_dim_dens2, conf_lrn_rate):
        self.sess = sess
        self.conf_lrn_rate = conf_lrn_rate
        self.conf_dis_fac = 0.9
        self.output_action_num = output_action_num
        self.policy_model = self.set_policy_model(input_state_num, output_action_num, model_dim_dens1, model_dim_dens2)        
        self.set_optimizer()
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        self.action_prob_array = []
        self.action_index_array = []
        K.set_session(sess)
        
    def set_policy_model(self, input_state_num, output_action_num, model_dim_dens1, model_dim_dens2):
        # Build deep sequential model
        #  1st layer - LSTM sequential layer
        model = Sequential()
        ModSeq1 = Dense(model_dim_dens1, activation = 'relu',  input_shape = [input_state_num])
        model.add(ModSeq1)
        #  2nd layer - LSTM sequential layer        
        ModSeq2 = Dense(model_dim_dens2, activation = 'relu')
        model.add(ModSeq2)
        # 2nd layer - dense layer, softmax activation
        ModDens1 = Dense(output_action_num, activation = 'softmax')
        model.add(ModDens1)
        # print model structure
        model.summary()
        return model
    
    def set_optimizer(self):
        self.action_index_opt = tf.placeholder("float", [None, self.output_action_num])
        self.value_normdis_opt = tf.placeholder("float", [None, ])
        self.policy_actionprob_tf = tf.reduce_sum(self.action_index_opt*self.policy_model.output,1)
        self.policy_log_tf = tf.log(self.policy_actionprob_tf)
        self.policy_obj_tf = tf.multiply(self.policy_log_tf, self.value_normdis_opt)
        cross_entropy = -K.sum(self.policy_obj_tf)        
        gradient_tf = tf.gradients(cross_entropy, self.policy_model.trainable_weights)
        grad = zip(gradient_tf, self.policy_model.trainable_weights)
        self.run_opt = tf.train.AdamOptimizer(self.conf_lrn_rate).apply_gradients(grad)        
        self.sess.run(tf.global_variables_initializer())
    
    def train_model(self, state_array, action_array, reward_array):
        value_normdis = self.calc_norm_values(reward_array)
        self.policy_actionprob_value, self.policy_log_value, self.policy_obj_value, optimal_result = self.sess.run(
                [self.policy_actionprob_tf, self.policy_log_tf, self.policy_obj_tf, self.run_opt],
                feed_dict = {
                        self.policy_model.input: state_array,
                        self.action_index_opt: action_array,
                        self.value_normdis_opt: value_normdis})
        return optimal_result
        
    def get_action(self, state):
        self.model_out = self.policy_model.predict(state)[0]            
        action_index = np.random.choice(self.output_action_num, 1, p=self.model_out)[0]
        return action_index
    
    def store_sample(self, state, action_index, action_index_set, action_prob, reward):
        self.state_array.append(state[0])
        self.action_array.append(action_index)
        self.action_index_array.append(action_index_set)
        self.reward_array.append(reward)
        self.action_prob_array.append(action_prob)

    def reset_sample(self):
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        self.action_prob_array = []
        self.action_index_array = []
        
    def calc_norm_values(self, reward_array):
        # Calculate accumulated rewards with discount factor
        values_dis = np.zeros_like(reward_array)
        sum_val = 0        
        for step_index in reversed(range(0, len(reward_array))):
            sum_val = sum_val * self.conf_dis_fac + reward_array[step_index]
            values_dis[step_index] = sum_val
        values_dis = np.float32(values_dis)
#         Calculate normalized value array        
        values_norm = values_dis - np.mean(values_dis)
        values_norm = values_norm / (np.std(values_norm)+0.000001)
#        values_norm = (values_dis)/10
        values_norm = np.reshape(values_norm, -1)
        self.values_norm = values_norm        
        self.values_dis = values_dis
        return values_norm
    
#%% Polic gradient - Gaussian
class agent_mc_gaussian:
    def __init__(self, sess, input_state_num, model_dim_dens1, model_dim_dens2, conf_lrn_rate):
        self.sess = sess
        self.conf_lrn_rate = conf_lrn_rate
        self.conf_dis_fac = 0.9
        self.conf_std = 0.1        
        self.policy_model, self.policy_model_weights, self.policy_model_input = self.set_policy_model(input_state_num, model_dim_dens1, model_dim_dens2)
        self.set_optimizer()
        self.state_array = []
        self.action_array = []
        self.reward_array = []
        self.action_prob_array = []  
        self.set_gradient()
        K.set_session(sess)
        
  
    def set_policy_model(self, input_state_num, model_dim_dens1, model_dim_dens2):
        # Build deep sequential model
        #  1st layer - LSTM sequential layer
        model = Sequential()
        ModSeq1 = Dense(model_dim_dens1, activation = 'relu',  input_shape = [input_state_num])
        model.add(ModSeq1)
        #  2nd layer - LSTM sequential layer        
        ModSeq2 = Dense(model_dim_dens2, activation = 'relu')
        model.add(ModSeq2)
        # 2nd layer - dense layer, softmax activation
        ModDens1 = Dense(1, activation = 'tanh')
        model.add(ModDens1)
        # print model structure
        model.summary()
        return model, model.trainable_weights, model.input

    def set_optimizer(self):
        # Determine place holder for loss function arguments        
        self.action_value = tf.placeholder(tf.float32,[None, 1])
        self.reward_normdis = tf.placeholder(tf.float32,[None, 1])
        gaussian_log = tf.square((self.action_value - self.policy_model.output))/(self.conf_std**2)
        gaussian_loss = -K.sum(tf.multiply(gaussian_log, self.reward_normdis))
#        gradient_gaussian_log = tf.gradients(gaussian_log, self.policy_model_weights)*self.reward_normdis
        gradient_model = tf.gradients(gaussian_loss, self.policy_model_weights)
        grads = zip(gradient_model, self.policy_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.conf_lrn_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        
    def set_gradient(self,):
        self.action_value_grad = tf.placeholder(tf.float32,[None, 1])
        self.reward_normdis_grad = tf.placeholder(tf.float32,[None, 1])
        self.gaussian_log_tf = -tf.square((self.action_value_grad - self.policy_model.output))/(self.conf_std**2)
        self.gaussian_object_tf = tf.multiply(self.gaussian_log_tf, self.reward_normdis_grad)
        self.gaussian_log_grad_tf = tf.gradients(self.gaussian_object_tf, self.policy_model.trainable_weights)
        print('grad_set')
    
    def train_weight(self,state_array, action_array, reward_array):
        reward_normdis = self.calc_norm_values(reward_array)
        self.reward_sum = np.sum(reward_normdis)
        self.gaussian_log_value, self.gaussian_object_value, self.gaussian_grad_value = self.sess.run(
                [self.gaussian_log_tf, self.gaussian_object_tf, self.gaussian_log_grad_tf],
                feed_dict = {
                self.action_value_grad: action_array,
                self.policy_model.input: state_array,
                self.reward_normdis_grad: reward_normdis})
        weight_values = self.policy_model.get_weights()
        for weight_index in range(len(weight_values)):
            delta_values = self.conf_lrn_rate*self.gaussian_grad_value[weight_index]
            weight_values[weight_index] += delta_values
        self.policy_model.set_weights(weight_values)
            
    def train(self, state_array, action_array, reward_array):
        reward_normdis = self.calc_norm_values(reward_array)
        self.sess.run(self.optimize, feed_dict = {
            self.policy_model.input: state_array,
            self.action_value: action_array,
            self.reward_normdis: reward_normdis
        })
    
    def clac_loss(self, state_array, action_array, reward_array):
        self.policy_model_output = self.policy_model.predict(state_array)
        self.reward_normdis_loss = self.calc_norm_values(reward_array)
        self.gaussian_log_loss = np.square((action_array - self.policy_model_output))/(self.conf_std**2)
        cross_entropy = -self.gaussian_log_loss * self.reward_normdis_loss
        loss = np.sum(cross_entropy)
        self.cross_entropy = cross_entropy
        self.loss = loss
        return loss
        
    def get_action(self, state):
        # Predict mean value of gaussian
        self.model_out = self.policy_model.predict(state)[0]
        action_index = np.random.normal(self.model_out, self.conf_std)
        return action_index
    
    def store_sample(self, state, action_index, reward):
        self.state_array.append(state[0])
        self.action_array.append(action_index)
        self.reward_array.append(reward)        
    
    def calc_norm_values(self, reward_array):
        # Calculate accumulated rewards with discount factor
        values_dis = np.zeros_like(reward_array)
        sum_val = 0        
        for step_index in reversed(range(0, len(reward_array))):
            sum_val = sum_val * self.conf_dis_fac + reward_array[step_index]
#            sum_val = sum_val * 0.99 + reward_array[step_index]
            values_dis[step_index] = sum_val
        values_dis = np.float32(values_dis)
        values_norm = values_dis - np.mean(values_dis)
        values_norm = values_norm / (np.std(values_norm)+0.000001)
#        values_norm = values_dis/10
        values_norm = np.reshape(values_norm, [-1, 1])
        self.values_norm = values_norm        
        self.values_dis = values_dis
        return values_norm
#%% Dqrn
class DdqrnAgent:

    def __init__(self, state_num, sequence_num, action_dim, lrn_rate, agent_config = None):
        # Define state, action, sequence size for model input, output
        self.state_num = state_num
        self.action_dim = action_dim
        self.sequence_num = sequence_num   
        self.lrn_rate = lrn_rate
        # Create replay memory
        self.memory = ReplayMemory()        
        # Create main model and target model
        self.model = None
        self.target_model = None
        # Define hyper parameters
        print('==== Hyper param list: dis_fac, epsilon_init, epsilon_term, batch_size, target_up_freq, explore_dn_freq ====')
        self.set_hyper_param(agent_config)
        self.set_agent_model(state_num, sequence_num, action_dim, lrn_rate)
        self.q_step = 0
    
    def set_agent_model(self, state_num, sequence_num, action_dim, lrn_rate):
        self.agent_model_config = NetworkDrqn()
        self.model = self.agent_model_config.model_def(state_num, sequence_num, action_dim, lrn_rate)
        self.target_model = self.agent_model_config.model_def(state_num, sequence_num, action_dim, lrn_rate)
        
    def set_hyper_param(self, agent_config):
        if agent_config == None:
            self.dis_fac = 0.98            
            self.epsilon_init = 0.5
            self.epsilon_term = 0.1                   
            self.batch_size = 32 
            self.target_up_freq = 200
            self.explore_dn_freq = 300
        else:
            self.dis_fac = agent_config['dis_fac']            
            self.epsilon_init = agent_config['epsilon_init']
            self.epsilon_term = agent_config['epsilon_term']
            self.batch_size = agent_config['batch_size']
            self.target_up_freq = agent_config['target_up_freq']
            self.explore_dn_freq = agent_config['explore_dn_freq']
        self.epsilon = self.epsilon_init
        self.lrn_num = 0
        self.q_current = 0
        self.q_target = 0
        
    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        if (self.lrn_num+1)%self.target_up_freq == 0:
            print('!!! target model update !!!')
            self.target_model.set_weights(self.model.get_weights())
        
    def update_epsilon(self):
        "Update explore parameter"
        if (self.lrn_num+1)%self.explore_dn_freq == 0:
            self.epsilon = self.epsilon - 0.0001
            if self.epsilon <= self.epsilon_term:
                print('!!! explore over !!!')
                self.epsilon = self.epsilon_term            

    def get_action(self, state_sequence):
        """
        Get action from model using epsilon-greedy policy
        """
        self.update_epsilon()
        q = self.model.predict(state_sequence)
        self.q_step = q
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_dim)            
        else:
            action_idx = np.argmax(q)            
        return action_idx
    
    def train_from_replay(self):
        # Get training sample set
        sample_traces, training_state = self.memory.get_sample_training_set(self.batch_size, self.sequence_num) #batch_dize x sequence_num x 4        
        self.update_target_model()
        if training_state == 'observe':
            self.flag_train_state = 0
            q_max = 0
            loss = 0
        else:
            self.flag_train_state = 1
            # Shape (batch_size, sequence_num, state_num)
            # state_input_current = np.zeros(((self.batch_size,) + self.state_num)) # Model input array of current state
            # state_input_next = np.zeros(((self.batch_size,) + self.state_num)) # Model input array of next state
            
            state_input_current = np.zeros((self.batch_size, self.sequence_num, self.state_num))
            state_input_next =  np.zeros((self.batch_size, self.sequence_num, self.state_num))
            action = np.zeros((self.batch_size, self.sequence_num)) # 32x8
            reward = np.zeros((self.batch_size, self.sequence_num))
    
            for i in range(self.batch_size):
                for j in range(self.sequence_num):
                    state_input_current[i,j,:] = sample_traces[i][j][0]
                    action[i,j] = sample_traces[i][j][1]
                    reward[i,j] = sample_traces[i][j][2]
                    state_input_next[i,j,:] = sample_traces[i][j][3]
    
            """
            # Use all traces for training
            # Size (batch_size, sequence_num, action_dim)
            # target = self.model.predict(update_input) # 32x8x3
            # target_val = self.model.predict(update_target) # 32x8x3
            # for i in range(self.batch_size):
            #     for j in range(self.sequence_num):
            #         a = np.argmax(target_val[i][j])
            #         target[i][j][int(action[i][j])] = reward[i][j] + self.dis_fac * (target_val[i][j][a])
            """
    
            # Only use the last trace for training
            q_current = self.model.predict(state_input_current) # 32x3
            q_target = q_current
            
            q_next = self.model.predict(state_input_next) # 32x3
            # q_next_model = self.target_model.predict(state_input_next) # 32x3
            
            for i in range(self.batch_size):
                a = np.argmax(q_next[i])
                q_target[i][int(action[i][-1])] = reward[i][-1] + self.dis_fac * (q_next[i][a])
            
            loss = self.model.train_on_batch(state_input_current, q_target)
            q_max = np.max(q_target[-1,-1])
            self.lrn_num = self.lrn_num + 1
            self.q_current = q_current
            self.q_target = q_target            
        return q_max, loss

class NetworkDrqn:
    def __init__(self, lstm_activation = 'tanh', dense_activation = 'linear', lstm_feanum = 265):
        self.lstm_activation = lstm_activation
        self.dense_activation = dense_activation
        self.lstm_feanum = lstm_feanum
        
    def model_def(self, input_state_num, input_sequence_num, output_dim, learning_rate):
        # Use last trace for training
        input_shape = (input_sequence_num, input_state_num)
        model = Sequential()
        self.model_seq1 = LSTM(self.lstm_feanum,  return_sequences=False, activation = self.lstm_activation, input_shape = input_shape)
        model.add(self.model_seq1)
        self.model_output = Dense(output_dim=output_dim, activation='linear')
        model.add(self.model_output)
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)
        model.summary()
        return model

class ReplayMemory:
    """
    Memory management class for model training
    """
    def __init__(self, buffer_size=1000):
        
        self.buffer = []
        self.episode_experience = []
        self.buffer_size = buffer_size        
        
    def store_sample(self, state, action_step, reward_step, state_target):
        sample_data = [state, action_step, reward_step, state_target]
        self.episode_experience.append(sample_data)
    
    def add_episode_buffer(self,):
        if len(self.buffer) + 1 >= self.buffer_size:
#            print('!!! buffer is full !!!')
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []            
        self.buffer.append(self.episode_experience)
        self.episode_experience = []

    def get_sample_training_set(self, batch_size, sequence_num):
        if len(self.buffer)<= batch_size:
            print('!!! observation require !!!')
            sampled_training_set = []
            training_state = 'observe'
        else:
            sampled_episodes = random.sample(self.buffer, batch_size)
            sampled_training_set = []
            for episode in sampled_episodes:
                point = np.random.randint(0, len(episode)+1-sequence_num)
                sampled_training_set.append(episode[point:point+sequence_num])
            sampled_training_set = np.array(sampled_training_set)
            training_state = 'train'
        return sampled_training_set, training_state