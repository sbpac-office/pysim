# -*- coding: utf-8 -*-
"""
Application: Smart regenerative braking based on reinforment learning
======================================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Environment - Reward calculation

Update
~~~~~~~~~~~~~
* [19/02/22] - Initial draft design
"""
import numpy as np
#%% Environment class
class EnvBrake:
    def __init__(self):
        self.reward_velocity = 0
        self.reward_drv = 0
        self.reward_safty = 0
        self.soc_old = 0
        self.set_reward_coef()
        self.reg_flag = 'reg_on'

    def set_reward_coef(self, reward_conf_acc = 0.5, reward_val_drv = -10, reward_conf_drvint_speed = 4,
                   reward_conf_loc_across_dis = -3, reward_conf_loc_across_spd = 1,  reward_val_location_across = -300,
                   reward_conf_loc_behind_dis = 5, reward_conf_loc_behind_spd = 16, reward_val_location_behind = -200):
        self.conf_drv_int_speed = reward_conf_drvint_speed
        self.reward_conf_acc = reward_conf_acc
        self.reward_val_drv = reward_val_drv
        self.reward_val_location_behind = reward_val_location_behind
        self.reward_val_location_across = reward_val_location_across
        self.reward_conf_loc_across_dis = reward_conf_loc_across_dis
        self.reward_conf_loc_across_spd = reward_conf_loc_across_spd
        self.reward_conf_loc_behind_dis = reward_conf_loc_behind_dis
        self.reward_conf_loc_behind_spd = reward_conf_loc_behind_spd

    def get_reward_drv(self, veh_vel_set, veh_vel):
        if abs(veh_vel_set-veh_vel) >= self.conf_drv_int_speed:
            self.reg_flag = 'drv_int'
            self.reward_drv = self.reward_val_drv
        else:
            self.reward_drv = 0
        return self.reward_drv

    def get_reward_saf(self, data_vel, data_dis):
        behind_cri_dis = data_vel*self.reward_conf_loc_behind_spd + self.reward_conf_loc_behind_dis
        across_cri_dis = data_vel*self.reward_conf_loc_across_spd + self.reward_conf_loc_across_dis
        if data_dis <= across_cri_dis:
            self.reward_safty = self.reward_val_location_across
            self.reg_flag = 'saf'
        elif data_dis >= behind_cri_dis:
            self.reward_safty = self.reward_val_location_behind
            self.reg_flag = 'saf'
        else:
            self.reward_safty = 0
        self.behind_cri_dis = behind_cri_dis
        self.across_cri_dis = across_cri_dis
        return self.reward_safty

#    def get_reward_vel(self, velocity_guide_line, velocity_veh):
#        self.reward_velocity = 0.01 - self.reward_conf_vel*abs(velocity_guide_line - velocity_veh)
#        self.reg_flag = 'reg_on'
#        return self.reward_velocity

    def get_reward_acc(self, acc_guide, veh_acc):
        self.reward_acc = 0.1 - self.reward_conf_acc*abs(acc_guide - veh_acc)
        self.reg_flag = 'reg_on'
        return self.reward_acc

    def get_reward(self, acc_guide, veh_acc, veh_vel_set, veh_vel ,data_dis):
        r_acc = self.get_reward_acc(acc_guide, veh_acc)
        r_saf = self.get_reward_saf(veh_vel, data_dis)
        r_drv = self.get_reward_drv(veh_vel_set, veh_vel)
        self.reward_sum = r_acc + r_saf + r_drv
        return self.reward_sum
#%% Environment driving case
class EnvRegen:
    ''' ============ environment class for rl regen ===========================
     1. Determine reward of regen control
     * Driving: Reward for similarity with driving data
     * Model: Reward for similarity with driver model
     * Safety: Reward for safety by collison
     * Energy: Reward for regnerative energy

     2. Determine input state conditions
     * Vehicle state: Vel, Acc, Motspd
     * Object state: Prevel, Relvel, Reldis, Refacc
     * Model state: Time, Brksection
    '''
    def __init__(self, soc):

        self.set_reward_coef_init()
        self.set_state_range()
        self.soc = soc
        print('Import rl regen environment')
    " Environment for reward "
    def set_reward_coef_init(self,):
        "Set configurable parameters to reward calculation"
        self.set_coef_driving()
        self.set_coef_model()
        self.set_coef_safety()
        self.set_coef_energy()
        self.rv_driving = 0
        self.rv_model = 0
        self.rv_safety = 0
        self.rv_energy = 0

    def set_coef_driving(self, conf_drv_fac_acc = 1, conf_drv_fac_vel = 0.5, conf_drv_fac_dis = 0.5):
        "Set driving reward configuration"
        "reward = -conf_factor * error(driving_data, control_result)"
        self.conf_drv_fac_acc = conf_drv_fac_acc
        self.conf_drv_fac_vel = conf_drv_fac_vel
        self.conf_drv_fac_dis = conf_drv_fac_dis

    def set_coef_model(self, conf_mod_fac_acc = 0.5, conf_mod_fac_vel = 0.25, conf_mod_fac_dis = 0.25):
        "Set model reward configuration"
        "reward = -conf_factor * error(model_data, control_result)"
        self.conf_mod_fac_acc = conf_mod_fac_acc
        self.conf_mod_fac_vel = conf_mod_fac_vel
        self.conf_mod_fac_dis = conf_mod_fac_dis

    def set_coef_safety(self, conf_saf_fac_cri = 100, conf_saf_dis = 3):
        "Set safety reward configuration - collison occur, termination distance"
        self.conf_saf_fac_cri = conf_saf_fac_cri
        self.conf_saf_dis = conf_saf_dis

    def set_coef_energy(self, conf_energy_fac = 0.1):
        "Set energy reward configuration"
        "reward = conf_factor * SOC variation"
        self.conf_energy_fac = conf_energy_fac

    def get_reward(self, driving_data, model_data, control_result, soc = 0):
        "Reward sum"
        r_drv = self.get_reward_drving(driving_data, control_result)
        r_mod = self.get_reward_model(model_data, control_result)
        r_saf = self.get_reward_safety(control_result)
        r_eng = self.get_reward_energy(soc)
        self.rv_sum = r_drv + r_mod + r_saf + r_eng
        return self.rv_sum

    def get_reward_drving(self, driving_data, control_result):
        "Calculate driving reward"
        err_acc = abs(driving_data['acc'] - control_result['acc'])
        err_vel = abs(driving_data['vel'] - control_result['vel'])
        err_dis = abs(driving_data['reldis'] - control_result['reldis'])
        self.rv_driving_acc = -self.conf_drv_fac_acc * err_acc
        self.rv_driving_vel = -self.conf_drv_fac_vel * err_vel
        self.rv_driving_dis = -self.conf_drv_fac_dis * err_dis
        self.rv_driving = self.rv_driving_acc + self.rv_driving_vel + self.rv_driving_dis
        return self.rv_driving

    def get_reward_model(self, model_data, control_result):
        "Calculate model reward"
        err_acc = abs(model_data['acc'] - control_result['acc'])
        err_vel = abs(model_data['vel'] - control_result['vel'])
        err_dis = abs(model_data['reldis'] - control_result['reldis'])
        self.rv_model_acc = -self.conf_mod_fac_acc * err_acc
        self.rv_model_vel = -self.conf_mod_fac_vel * err_vel
        self.rv_model_dis = -self.conf_mod_fac_dis * err_dis
        self.rv_model = self.rv_model_acc + self.rv_model_vel + self.rv_model_dis
        return self.rv_model

    def get_reward_safety(self, control_result):
        "Check collison"
        reldis = control_result['reldis']
        if reldis <= self.conf_saf_dis:
            self.rv_safety = -self.conf_saf_fac_cri/10
        elif reldis == 0:
            self.rv_safety = -self.conf_saf_fac_cri
        else:
            self.rv_safety = 0
        return self.rv_safety

    def get_reward_energy(self, soc):
        "Calculate reward for energy efficiency"
        soc_delta = soc - self.soc
        self.rv_energy = soc_delta * self.conf_energy_fac
        self.soc = soc
        return self.rv_energy

    " Environment for state "
    def set_state_range(self, ):
        "Configure min max range of states"
        self.min_max_config = {}
        self.set_state_range_veh()
        self.set_state_range_obj()
        self.set_state_range_mod()

    def set_state_range_veh(self, veh_acc_min = -5, veh_acc_max = 0, veh_vel_min = 0, veh_vel_max = 120):
        "Configure min max range of states"
        self.min_max_config['veh_acc'] = [veh_acc_min, veh_acc_max]
        self.min_max_config['veh_vel'] = [veh_vel_min, veh_vel_max]

    def set_state_range_obj(self, obj_relvel_min = -10, obj_relvel_max = 5, obj_reldis_min = 0, obj_reldis_max = 200,
                            obj_prevel_min = 0, obj_prevel_max = 120, obj_refacc_min = -5, obj_refacc_max = 2):
        "Configure min max range of states"
        self.min_max_config['obj_relvel'] = [obj_relvel_min, obj_relvel_max]
        self.min_max_config['obj_reldis'] = [obj_reldis_min, obj_reldis_max]
        self.min_max_config['obj_prevel'] = [obj_prevel_min, obj_prevel_max]
        self.min_max_config['obj_refacc'] = [obj_refacc_min, obj_refacc_max]

    def set_state_range_mod(self, mod_time_min = 0, mod_time_max = 20, mod_section_min = 1, mod_section_max = 4):
        "Configure min max range of states"
        self.min_max_config['mod_time'] = [mod_time_min, mod_time_max]
        self.min_max_config['mod_section'] = [mod_section_min, mod_section_max]

    def get_norm_state_value(self, model_data, control_result):
        "Normalize each staes value"
        self.st_veh_acc = self.fcn_state_norm(control_result['acc'], self.min_max_config['veh_acc'])
        self.st_veh_vel = self.fcn_state_norm(control_result['vel'], self.min_max_config['veh_vel'])
        self.st_obj_relvel = self.fcn_state_norm(control_result['relvel'], self.min_max_config['obj_relvel'])
        self.st_obj_reldis = self.fcn_state_norm(control_result['reldis'], self.min_max_config['obj_reldis'])
        self.st_obj_prevel = self.fcn_state_norm(control_result['prevel'], self.min_max_config['obj_prevel'])
        self.st_obj_refacc = self.fcn_state_norm(control_result['accref'], self.min_max_config['obj_refacc'])
        self.st_mod_time = self.fcn_state_norm(model_data['time'], self.min_max_config['mod_time'])
        self.st_mod_section = self.fcn_state_norm(model_data['section'], self.min_max_config['mod_section'])
        state_array = np.array([self.st_veh_acc, self.st_veh_vel,
                                self.st_obj_relvel, self.st_obj_reldis, self.st_obj_prevel, self.st_obj_refacc,
                                self.st_mod_time, self.st_mod_section])
        self.state_array = state_array
        return state_array

    def fcn_state_norm(self, data, min_max_val):
        "Nomalization function for state"
        min_val = float(min_max_val[0])
        max_val = float(min_max_val[1])
        # Crip data
        data_lim = sorted((min_val, float(data), max_val))[1]
        # Normalization
        data_norm = (data_lim - min_val)/(max_val - min_val)
        return data_norm
