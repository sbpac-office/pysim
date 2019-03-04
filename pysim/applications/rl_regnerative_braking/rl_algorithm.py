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
from keras.layers import Dense, LSTM, Reshape, Input
from keras.optimizers import Adam, SGD, Nadam
from keras.models import Sequential, load_model
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
    
        
#%% Agent class_ policy gradient
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
    
#%% Agent class_ policy gradient
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


