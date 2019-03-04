# -*- coding: utf-8 -*-
"""
Application: Smart regenerative braking based on reinforment learning
======================================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Intelligent driver model

Update
~~~~~~~~~~~~~
* [19/02/22] - Initial draft design
"""
import numpy as np
import math
import os
import get_data
get_data.set_dir(os.path.abspath('.\driver_data'))
EffectProbIndex = get_data.load_mat('EffectiveProbIndex.mat')['ParLrn_ProbVec']


def EffectiveProbability(Index, IndexVectorArray, Std):
    "Calculation of effective probability"
    CoastIndexLength = np.size(IndexVectorArray);
    ProbVec = np.arange(CoastIndexLength, dtype = 'float' )
    # Gaussian distribution 
    for i in range(CoastIndexLength):
        VecVal = 1/(Std*math.sqrt(2*math.pi))*np.exp(-0.5*((Index - IndexVectorArray[i])/Std)**2)
        ProbVec[i,] = VecVal[0,0]
    # Normalization
    ProbSum = np.sum(ProbVec)
    EffProb = ProbVec/ProbSum
    return EffProb

class DecelStateRecog:    
    def __init__(self):
        self.drv_aps_in_pre = 0
        self.drv_bps_in_pre = 0
        self.acc_pedal_trans_val = 1
        self.brk_pedal_trans_val = 1
        self.stRegCtl = 'driving'
    
    def pedal_transition_machine(self, drv_aps_in, drv_bps_in):
        if (drv_aps_in >= self.acc_pedal_trans_val) and (self.drv_aps_in_pre < self.acc_pedal_trans_val):
            stDrvPedalTrns = 'acc on'
        elif (self.drv_aps_in_pre >= self.acc_pedal_trans_val) and (drv_aps_in < self.acc_pedal_trans_val):
            stDrvPedalTrns = 'acc off'
        elif (drv_bps_in >= self.brk_pedal_trans_val) and (self.drv_bps_in_pre < self.brk_pedal_trans_val):
            stDrvPedalTrns = 'brk on'
        elif (self.drv_bps_in_pre >= self.brk_pedal_trans_val) and (drv_bps_in < self.brk_pedal_trans_val):
            stDrvPedalTrns = 'brk off'
        else:
            stDrvPedalTrns = 'none'            
        self.drv_aps_in_pre = drv_aps_in
        self.drv_bps_in_pre = drv_bps_in
        return stDrvPedalTrns 
    
    # def regen_control_machine(self,stDrvPedalTrns):
    #     stRegCtl = self.stRegCtl
    #     if (stDrvPedalTrns == 'acc off') and stRegCtl == 'driving':
    #         stRegCtl = 'reg on'
    #     elif (stRegCtl == 'reg on') and ((stDrvPedalTrns == 'brk on') or (stDrvPedalTrns == 'acc on')):
    #         stRegCtl = 'drv int'
    #     elif stRegCtl == 'drv int':
    #         stRegCtl = 'driving'
    #     else:
    #         stRegCtl = 'driving'                
    #     self.stRegCtl = stRegCtl
    #     return stRegCtl
        
    def regen_control_machine(self, stDrvPedalTrns):
        stRegCtl = self.stRegCtl
        if (stDrvPedalTrns == 'acc off') and stRegCtl == 'driving':
            stRegCtl = 'reg on'
        elif (stRegCtl == 'reg on') and (stDrvPedalTrns == 'acc on'):
            stRegCtl = 'driving'            
        else:
            stRegCtl = self.stRegCtl              
        self.stRegCtl = stRegCtl
        return stRegCtl
    
class idm_acc_stop:
    def __init__(self, param_case):
        self.param_arrange(param_case)
        self.param_acc_diff_term = 0.01      
        self.acc_delta = 0        
        self.acc_delta_d = 0
        self.acc_delta_p = 0
        self.flag_idm_state = 0
#        print('Model set - '+str(param_case['DisCoast'])+'-'+str(param_case['DisInit']))
        pass
    
    def param_arrange(self, param_case):
        self.param_acc_cst = -0.3
        self.param_dis_cst = param_case['DisCoast']
        self.param_dis_init = param_case['DisInit']
        self.param_acc_ref_diff = param_case['AccRefDiff']
        self.param_acc_rat_adj = param_case['AccRatAdj']        
#        self.param_acc_adj = (param_case['AccRefInit']+param_case['AccRefDiff'])*param_case['AccRatAdj']
#        self.param_accslope_init = (param_case['AccInit'] - self.param_acc_adj)/param_case['TpDelta']/10
        self.param_acc_adj = param_case['AccAdj']
        self.param_accslope_init = -param_case['AccSlope']/10
        self.param_adj_gain = param_case['AdjGain']
        self.param_adj_gain_p = param_case['AdjGain_P']
        self.param_term_gain = param_case['TermGain']       
        self.param_term_gain_p = param_case['TermGain_P']       
        
    def state_def(self, stBrkStatePri, veh_data):        
        veh_acc = veh_data['acc']
        veh_acc_ref = veh_data['acc_ref']        
        veh_dis = veh_data['dis']
        veh_vel = veh_data['vel']
        
        if (veh_dis <= self.param_dis_init) and (veh_acc > self.param_acc_adj) and (self.flag_idm_state <= 2):
            stBrkState = 'Init'
            self.flag_idm_state = 2
            if stBrkStatePri == 'Cst':
                tmpParam_acc_ref_cst = veh_acc_ref
                self.param_acc_adj = (tmpParam_acc_ref_cst + self.param_acc_ref_diff)*self.param_acc_rat_adj       
                
        elif (veh_dis <= self.param_dis_cst) and (veh_acc > self.param_acc_adj) and (self.flag_idm_state <= 1):
            stBrkState = 'Cst'
            self.flag_idm_state = 1
            
        elif ((abs(veh_acc - veh_acc_ref) <= self.param_acc_diff_term) and (stBrkStatePri == 'Adj' or stBrkStatePri == 'Term')) or stBrkStatePri == 'Term':
            stBrkState = 'Term'
            self.flag_idm_state = 4
            if (veh_vel <= 0.001) or (veh_dis <= 0.001):
                stBrkState = 'None'
                
        elif (veh_acc <= self.param_acc_adj) or (self.flag_idm_state == 3):
            stBrkState = 'Adj'
            self.flag_idm_state = 3
            
        else:
            stBrkState = 'None'
            self.flag_idm_state = 0
            
        return stBrkState
    
    def acc_set_gen(self, stBrkState, veh_data):
        veh_acc_set = veh_data['acc_set']
        veh_acc_ref = veh_data['acc_ref']
        if stBrkState == 'Cst':
            acc_set = self.param_acc_cst
        elif stBrkState == 'Init':
            acc_set = veh_acc_set - self.param_accslope_init
        elif stBrkState == 'Adj':
            self.acc_delta = self.acc_delta + self.param_adj_gain*(veh_acc_ref - veh_acc_set)
            self.acc_delta_p = self.param_adj_gain_p*(veh_acc_ref - veh_acc_set)
            acc_set = veh_acc_set + self.acc_delta + self.acc_delta_p
        elif stBrkState == 'Term':
#            self.acc_delta_p = self.param_term_gain_p*(veh_acc_ref - veh_acc_set)
            self.acc_delta_p = self.param_adj_gain_p*(veh_acc_ref - veh_acc_set)
            self.acc_delta = self.acc_delta + self.param_term_gain*(veh_acc_ref - veh_acc_set)
            acc_set = veh_acc_set + self.acc_delta + self.acc_delta_p
        else:
            acc_set = 0
        return acc_set
    
    def state_decoding(self, stBrkState):
        if stBrkState == 'Cst':
            stStateNum = 1
        elif stBrkState == 'Init':
            stStateNum = 2
        elif stBrkState == 'Adj':
            stStateNum = 3
        elif stBrkState == 'Term':
            stStateNum = 4
        else:
            stStateNum = 5
        
        return stStateNum

class IdmAccCf:
    def __init__(self, DriverData):        
        self.LrnVec_RelDisInitDelta = EffectProbIndex['CoastRelDis'][0,0] - DriverData['LrnVec_Param_RelDisInit']
        self.LrnVec_RelDisAdjDelta = EffectProbIndex['InitRelDis'][0,0] - DriverData['LrnVec_Param_RelDisAdj'] 
        self.LrnVec_AccSlope = DriverData['LrnVec_Param_AccSlopeCf']
        self.param_active = {'RelDisInit':100., 'RelDisAdj':80. , 'AccSlope':-1.5, 'AccAdj':-2., 
                             'AccTerm':-4., 'AccMax':3., 'AccCst':-0.08, 'VelRat':0.8, 'DisAdj':0., 
                             'GainAdj':0.05, 'GainTerm':0.5}
        
        self.param_active['DisAdjDelta'] = 0
        
        
        self.mod_profile = {'acc':0., 'acc_ref':0., 'vel':0., 'vel_ref':0., 'reldis':0., 'dis_eff':0.}
        self.stBrkState = 'None'
        self.flag_idm_run = 'off'
        self.flag_idm_state = 0
        print('Import idm model')
        pass
    
    def param_active_coast(self, veh_data_reldis_coast):
        # Activate initial rel dis parameter
        EffProb_InitRelDis = EffectiveProbability(veh_data_reldis_coast, EffectProbIndex['CoastRelDis'][0,0], EffectProbIndex['StdCoastRelDis'][0,0])        
        self.param_active['RelDisInitDelta'] = np.sum(EffProb_InitRelDis*self.LrnVec_RelDisInitDelta[:,0])
        self.param_active['RelDisInit'] = veh_data_reldis_coast - self.param_active['RelDisInitDelta']
        
        # Activate adjust rel dis parameter
        EffProb_AdjRelDis =  EffectiveProbability(self.param_active['RelDisInit'] , EffectProbIndex['InitRelDis'][0,0], EffectProbIndex['StdInitRelDis'][0,0])        
        self.param_active['RelDisAdjDelta'] = np.sum(EffProb_AdjRelDis*self.LrnVec_RelDisAdjDelta[:,0])
        self.param_active['RelDisAdj'] = self.param_active['RelDisInit'] - self.param_active['RelDisAdjDelta']
        
    def param_active_init(self, veh_data_reldis_init, veh_data_vel_init, mod_vel_ref_init):
        InitAccIndex = veh_data_vel_init**2/veh_data_reldis_init
        EffProb_AccSlopeInit = EffectiveProbability(InitAccIndex, EffectProbIndex['InitIndexCf'][0,0], EffectProbIndex['StdAcc'][0,0])
        self.param_active['AccSlope'] = np.sum(EffProb_AccSlopeInit*self.LrnVec_AccSlope[:,0])
        self.param_active['VelRat'] = veh_data_vel_init/mod_vel_ref_init
    
    def param_active_adj(self, veh_data_acc_adj):
        self.param_active['AccAdj'] = veh_data_acc_adj
        self.param_active['DisAdj'] = 0
        
    def param_active_term(self, veh_data_acc_term):
        self.param_active['AccTerm'] = veh_data_acc_term
        self.param_active['DisAdj'] = 0
        
    def modprof_init(self, veh_data, preveh_vel):
        self.mod_profile['acc'] = veh_data['acc']
        self.mod_profile['acc_ref'] = -0.5*(preveh_vel**2 - veh_data['vel']**2)/veh_data['reldis']
        self.mod_profile['vel'] = veh_data['vel']
        self.mod_profile['vel_ref'] = 0
        self.mod_profile['reldis'] = veh_data['reldis']
        self.mod_profile['dis_eff'] = 0
            
    def state_def(self, mod_profile, stBrkStatePri, stPedalTrns, veh_data, preveh_vel):        
        [acc, acc_ref, vel, vel_ref, reldis, dis_eff] = self.fcn_mod_profile_set(mod_profile)
        
        if self.flag_idm_run == 'off':
            if stPedalTrns == 'acc off':
                self.flag_idm_run = 'on'
                self.stBrkState = 'Cst'
                self.modprof_init(veh_data, preveh_vel)  
                [acc, acc_ref, vel, vel_ref, reldis, dis_eff] = self.fcn_mod_profile_set(self.mod_profile)
                self.param_active_coast(reldis)
                # print('RelDis: ', reldis, 'RelDisInit: ', self.param_active['RelDisInit'],'RelDisAdj: ', self.param_active['RelDisAdj'])
            else:
                self.flag_idm_run = 'off'
        else: 
            if stPedalTrns == 'acc on':
                self.flag_idm_run = 'off'
                self.stBrkState = 'None'
            else:
                self.flag_idm_run = 'on'
                
        if (self.flag_idm_run == 'on') and (vel >= preveh_vel):
            if (reldis <= self.param_active['RelDisInit']) and (reldis > self.param_active['RelDisAdj']) and (acc >= acc_ref) and (self.flag_idm_state <= 2):
                stBrkState = 'Init'
                self.flag_idm_state = 2
                if stBrkStatePri == 'Cst':
                    self.param_active_init(reldis, vel, vel_ref)                    
            elif (reldis <= self.param_active['RelDisAdj']) and (acc >= acc_ref) and ((self.flag_idm_state == 2) or (self.flag_idm_state == 3)):
                stBrkState = 'Adj'
                self.flag_idm_state = 3
                if stBrkStatePri == 'Init':
                    self.param_active_adj(acc)                    
            elif (self.flag_idm_run == 'on') and (self.flag_idm_state >= 2):
                stBrkState = 'Term'
                self.flag_idm_state = 4
                if (stBrkStatePri == 'Init') or (stBrkStatePri == 'Adj'):
                    self.param_active_term(acc)
            else:
                stBrkState = 'Cst'
                self.flag_idm_state = 1
        else:
            stBrkState = 'None'
            self.flag_idm_state = 0
                  
        return stBrkState
       
    def profile_calculation(self, stBrkState, mod_profile, preveh_vel):
        [acc, acc_ref, vel, vel_ref, reldis, dis_eff] = self.fcn_mod_profile_set(mod_profile)
        
        acc_ref_next = self.fcn_acc_ref_calc(vel, preveh_vel, reldis)
        if stBrkState == 'Cst':
            vel_ref_next = vel/pow((1 - self.param_active['AccCst']/self.param_active['AccMax']), 0.25)
            dis_eff_next = 0
        elif stBrkState == 'Init':            
            self.param_active['VelRat'] = pow( ( pow(self.param_active['VelRat'],4) - self.param_active['AccSlope']/self.param_active['AccMax']*0.1 ) , 0.25)
            vel_ref_next = vel/self.param_active['VelRat']
            dis_eff_next = 0
        elif stBrkState == 'Adj':
            vel_ref_next = vel/pow( -1*self.param_active['AccAdj']/self.param_active['AccMax'], 0.25) 
            self.param_active['DisAdjDelta'] = (acc - acc_ref)*self.param_active['GainAdj']/self.param_active['AccMax']
            self.param_active['DisAdj'] = self.param_active['DisAdj'] + self.param_active['DisAdjDelta']
            dis_eff_next = reldis* pow(self.param_active['DisAdj']+1,0.5)
        elif stBrkState == 'Term':
            vel_ref_next = vel/pow(-1*self.param_active['AccTerm']/self.param_active['AccMax'], 0.25)
            self.param_active['DisAdjDelta'] = (acc - acc_ref)*self.param_active['GainTerm']/self.param_active['AccMax']
            self.param_active['DisAdj'] = self.param_active['DisAdj'] + self.param_active['DisAdjDelta']
            dis_eff_next = reldis* pow(self.param_active['DisAdj']+1,0.5)
        else:
            acc_ref_next = 0
            vel_ref_next = vel
            dis_eff_next = 0
        mod_profile['acc_ref'] = acc_ref_next
        mod_profile['vel_ref'] = vel_ref_next
        mod_profile['dis_eff'] = dis_eff_next
            
        return mod_profile
    
    def acc_update(self, mod_profile, preveh_vel):        
        [acc, acc_ref, vel, vel_ref, reldis, dis_eff] = self.fcn_mod_profile_set(mod_profile)
                
        acc_next = self.param_active['AccMax']*(1 - pow((vel/vel_ref), 4) - pow((dis_eff/reldis), 2))
        vel_next = vel + acc_next*0.1
        relvel_next = preveh_vel - vel_next
        reldis_next = sorted((0.5, reldis + relvel_next*0.1, 100))[1]
        
        if self.flag_idm_run == 'on':
            mod_profile['acc'] = acc_next
            mod_profile['vel'] = vel_next
            mod_profile['reldis'] = reldis_next        
        else:
            mod_profile['acc'] = 0
            mod_profile['vel'] = 0
            mod_profile['reldis'] = 0        
        return mod_profile
    
    def profile_update(self, stBrkState, mod_profile_old, preveh_vel):
        mod_profile_mid = self.profile_calculation(stBrkState, mod_profile_old, preveh_vel)
        mod_profile_new = self.acc_update(mod_profile_mid, preveh_vel)
        return mod_profile_new
    
    def prediction(self, stPedalTrns, veh_data, preveh_vel):        
        self.stBrkState = self.state_def(self.mod_profile, self.stBrkState, stPedalTrns, veh_data, preveh_vel)
        self.mod_profile = self.profile_update(self.stBrkState, self.mod_profile, preveh_vel)        
    
    def fcn_acc_ref_calc(self, vel, preveh_vel, reldis):
        acc_ref = 0.5*(preveh_vel**2 - vel**2)/reldis
        return acc_ref                
    
    def fcn_mod_profile_set(self, mod_profile):
        acc = mod_profile['acc']
        acc_ref = mod_profile['acc_ref']        
        vel = mod_profile['vel']        
        vel_ref =  sorted((0.0001, mod_profile['vel_ref'], 100.))[1]
        reldis = sorted((0.00001, mod_profile['reldis'], 300.))[1]
        dis_eff =  mod_profile['dis_eff']    
        return acc, acc_ref, vel, vel_ref, reldis, dis_eff
        
    def state_decoding(self, stBrkState):
        if stBrkState == 'Cst':
            stStateNum = 1
        elif stBrkState == 'Init':
            stStateNum = 2
        elif stBrkState == 'Adj':
            stStateNum = 3
        elif stBrkState == 'Term':
            stStateNum = 4
        else:
            stStateNum = 5
        
        return stStateNum
   