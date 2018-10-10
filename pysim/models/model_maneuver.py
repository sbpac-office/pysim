# -*- coding: utf-8 -*-
"""
Simulation model : Maneuver
==============================================================

Description
~~~~~~~~~~~~~
* Simulate driver behavior

Modules summary
~~~~~~~~~~~~~~~~~
* Driver module - driver characteristics
    * set_char - set parameters according to driver characteristics
        * set_driver_param - set parameters

* Behavior module - contain ``driver module`` and simulate driver's behavior
    * Driver_set - set driver control parameter from determined driver module
    * Maneuver_config - set maneuver configuration
    * Lon_behavior - determine longitudinal behavior
        * Static_state_recog - longitudinal state recognition for static objectives
        * Dynamic_state_recog - longitudinal state recognition for dynamic objectives
        * Lon_vel_set - determine longitudinal velocity set point
        * Lon_control - control acceleration and brake pedal for velocity set point
    * Lat_behavior - determine lateral behavior
        * Lateral_state_recog - lateral state recognition for road offset, heading angle
        * Lat_control - control steering for road offset, heading angle

Update
~~~~~~~~~~~~~
* [18/05/31] - Initial release - Kyunghan
* [18/06/05] - Modification of lon control - Kyunghan
"""
# import python lib modules
from math import pi
import numpy as np
# import package modules
from pysim.sub_util.sub_utilities import Calc_PrDis, Filt_LowPass
from pysim.sub_util.sub_type_def import type_pid_controller, type_drvstate, type_objective, type_hyst, type_trnsition_delay
# import config data modules

# simulation sampling time
Ts = 0.01
"""global vairable: simulation sampling timeself.

you can declare other sampling time in application as vairable ``Ts``

"""
class Mod_Driver:
    """
    * Driver module
    """
    def __init__(self):
        self.set_driver_char('Normal')

    def set_driver_char(self, DriverChar = 'Normal'):
        """Set driver parameter values according to characteristics

        Characteristics:
            * Normal
            * Aggressive
            * Defensive
        """
        if DriverChar == 'Normal':
            self.set_driver_param(P_gain_lon = 2, I_gain_lon = 0.5, D_gain_lon = 0, P_gain_lat = 0.001, I_gain_lat = 0.0001, D_gain_lat = 0, P_gain_yaw = 0.1, I_gain_yaw = 0.1, D_gain_yaw = 0, shift_time = 0.5, max_acc = 4)
        elif DriverChar == 'Aggressive':
            self.set_driver_param(P_gain_lon = 2, I_gain_lon = 0.5, D_gain_lon = 0, P_gain_lat = 0.001, I_gain_lat = 0.0001, D_gain_lat = 0, P_gain_yaw = 0.1, I_gain_yaw = 0.1, D_gain_yaw = 0, shift_time = 0.5, max_acc = 4)
        elif DriverChar == 'Defensive':
            self.set_driver_param(P_gain_lon = 2, I_gain_lon = 0.5, D_gain_lon = 0, P_gain_lat = 0.001, I_gain_lat = 0.0001, D_gain_lat = 0, P_gain_yaw = 0.1, I_gain_yaw = 0.1, D_gain_yaw = 0, shift_time = 0.5, max_acc = 4)
        else:
            print('Set the driver only = [''Normal'', ''Aggressive'', ''Defensive'']')
            self.set_driver_param(P_gain_lon = 2, I_gain_lon = 0.5, D_gain_lon = 0, P_gain_lat = 0.001, I_gain_lat = 0.0001, D_gain_lat = 0, P_gain_yaw = 0.1, I_gain_yaw = 0.1, D_gain_yaw = 0, shift_time = 0.5, max_acc = 4)

    def set_driver_param(self, P_gain_lon = 2, I_gain_lon = 0.5, D_gain_lon = 0, P_gain_lat = 0.001, I_gain_lat = 0.0001, D_gain_lat = 0, P_gain_yaw = 0.1, I_gain_yaw = 0.1, D_gain_yaw = 0, shift_time = 0.5, max_acc = 4):
        """Set driver parameter values

        Parameters:
            * PID gains for lon control
            * PID gains for lat offset control
            * PID gains for yaw control
            * Shift time
            * Maximum acceleration value
        """
        self.P_gain_lon = P_gain_lon; self.I_gain_lon = I_gain_lon; self.D_gain_lon = D_gain_lon
        self.P_gain_lat = P_gain_lat; self.I_gain_lat = I_gain_lat; self.D_gain_lat = D_gain_lat
        self.P_gain_yaw = P_gain_yaw; self.I_gain_yaw = I_gain_yaw; self.D_gain_yaw = D_gain_yaw
        self.shift_time = shift_time; self.max_acc = max_acc

class Mod_Behavior:
    """
    * Behavior module: Set the ``driver`` model when initialization
    """
    def __init__(self, Driver):
        self.stStaticList = type_drvstate()
        self.stDynamicList = type_drvstate()
        self.stStatic = type_drvstate()
        self.stDynamic = type_drvstate()
        self.Maneuver_config()
        self.Drver_set(Driver)
        self.Ts_Loc = globals()['Ts']
        self.u_acc = 0
        self.u_brk = 0
        self.u_steer = 0
        self.u_steer_offset = 0
        self.u_steer_yaw = 0
        self.veh_speed_set = 0
        self.veh_speed_set_dynamic = 0
        self.veh_speed_set_static = 0



    def Drver_set(self, DriverSet):
        """Arrange driver parameters for behavior controller

        Define PID controller for velocity, offset, yaw

        Args:
            * DriverSet: driver parameter set
        """
        self.Driver = DriverSet
        self.Lon_Controller_acc = type_pid_controller(DriverSet.P_gain_lon, DriverSet.I_gain_lon, DriverSet.D_gain_lon)
        self.Lon_Controller_brk = type_pid_controller(DriverSet.P_gain_lon, DriverSet.I_gain_lon, DriverSet.D_gain_lon)
        self.Lat_Controller_offset = type_pid_controller(DriverSet.P_gain_lat, DriverSet.I_gain_lat, DriverSet.D_gain_lat)
        self.Lat_Controller_yaw = type_pid_controller(DriverSet.P_gain_yaw, DriverSet.I_gain_yaw, DriverSet.D_gain_yaw)
        self.stLonControl = 'idle'
        self.Filt_LonShiftTrnsDelay = type_trnsition_delay(DriverSet.shift_time)

    def Maneuver_config(self, cruise_speed_set = 15, mincv_speed_set = 5,
                        conf_curve_speed_set_curvcoef = 1000, conf_curve_speed_set_discoef = 0.01,
                        transition_dis = 20, forecast_dis = 200, cf_dis = 120, lat_off = 0.5,
                        filtnum_pedal = 0.1, filtnum_steer = 0.1, filtnum_spdset = 1):
        """Configure driver's maneuver

        Parameters:
            * Cruise speed set
            * Configurable parameters for static objectives
            * Filter values for driver's control input
        """
        self.conf_cruise_speed_set = cruise_speed_set
        self.conf_mincv_speed_set = mincv_speed_set
        self.conf_transition_dis = transition_dis
        self.conf_forecast_dis = forecast_dis
        self.conf_cf_dis = cf_dis
        self.conf_lat_off = lat_off
        self.conf_filtnum_spdset = filtnum_spdset
        self.conf_filtnum_pedal = filtnum_pedal
        self.conf_filtnum_steer = filtnum_steer
        self.conf_curve_speed_set_curvcoef = conf_curve_speed_set_curvcoef
        self.conf_curve_speed_set_discoef = conf_curve_speed_set_discoef

    def Static_state_recog(self,static_obj_in, road_len, veh_position_s):
        """Static state recognition for driving conditions

        Determine current longitudinal state for driving conditions

        Args:
            * static_obj_in: Static object information of driving route
            * road_len: Road length of driving route
            * veh_position_s: Current vehicle position on environment

        Returns:
            * stStatic: Static state
                * Tl (Traffic light): Distance to traffic light and traffic light state (red, green)
                * Curve: Distance to curve and curvature
                * Cruise: No specific object
        """
        # Define local state and objectives
        stStatic = type_drvstate()
        forecast_object = type_objective()
        transition_object = type_objective()
        # Determine map_index (forecasting, transition)
        if max(road_len) <= veh_position_s:
            print('========== Simulation is terminated!! ========= ')
            stStatic.set_state('Termination','None','None')
        else:
            tmp_cur_index = np.min(np.where(road_len >= veh_position_s))-1
            tmp_forecast_index = np.min(np.where(road_len >= (veh_position_s + self.conf_forecast_dis)))-1
            tmp_transition_index = np.min(np.where(road_len >= (veh_position_s + self.conf_transition_dis)))-1
            # Determine objectives from vehicle location to forecasting range
            for k in range(tmp_cur_index,tmp_forecast_index+1):
                forecast_object.merg_object(static_obj_in[k].object_class, static_obj_in[k].object_param, static_obj_in[k].object_loc_s)
            # Determine objectives from transition range to forecasting range
            for k in range(tmp_transition_index,tmp_forecast_index+1):
                transition_object.merg_object(static_obj_in[k].object_class, static_obj_in[k].object_param, static_obj_in[k].object_loc_s)
            if ('Tl' in forecast_object.object_class):
                tmp_Tl_index = forecast_object.object_class.index('Tl')
                tmp_Tl_param = forecast_object.object_param[tmp_Tl_index]
                tmp_Tl_loc = forecast_object.object_loc_s[tmp_Tl_index]
                if tmp_Tl_param == 'red':
                    stStatic.set_state('Tl_stop',tmp_Tl_param,tmp_Tl_loc - veh_position_s)
                else:
                    if 'Curve' in transition_object.object_class:
                        tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                        tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                        tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                        stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - veh_position_s)
                    else:
                        stStatic.set_state('Cruise')
            else:
                if 'Curve' in transition_object.object_class:
                    tmp_cv_index = np.where(np.array(transition_object.object_class) == 'Curve')[0]
                    tmp_cv_loc = np.mean(np.array(transition_object.object_loc_s)[tmp_cv_index])
                    tmp_cv_param = np.mean(np.array(transition_object.object_param)[tmp_cv_index])
                    stStatic.set_state('Curve',tmp_cv_param,tmp_cv_loc - veh_position_s)
                else:
                    stStatic.set_state('Cruise')

            self.stStaticList.add_state(stStatic.state, stStatic.state_param, stStatic.state_reldis)
            self.forecast_object = forecast_object
            self.transition_object = transition_object
            self.stStatic = stStatic
        return stStatic

    def Dynamic_state_recog(self, pre_veh_speed, pre_veh_reldis = 250):
        """Daynamic state recognition for driving conditions

        Determine current longitudinal state for preceding vehicle

        Args:
            * pre_veh_speed: Velocity of preceding vehicle [m/s]
            * pre_veh_reldis: Relative distance of preceding vehicle [m]

        Returns:
            * stDynamic: Dynamic state
                * Cf: Car-following state
                * Cruise: No specific object
        """
        stDynamic = type_drvstate()
        if pre_veh_reldis >= self.conf_cf_dis:
            stDynamic.set_state('Cruise')
        else:
            stDynamic.set_state('Cf', pre_veh_speed, pre_veh_reldis)
        self.stDynamicList.add_state(stDynamic.state, stDynamic.state_param, stDynamic.state_reldis)
        self.stDynamic = stDynamic
        return stDynamic

    def Lon_vel_set(self, stStatic, stDynamic):
        """Determine vehicle velocity set point according to longitudinal state

        Velocity set point algorithm::

            vel_set = min(vel_set_static, vel_set_dynamic)
                # vel_set_static
                if tmp_state_step_static == 'Cruise':
                    veh_speed_set_static = self.conf_cruise_speed_set
                elif tmp_state_step_static == 'Tl_stop':
                    veh_speed_set_static = self.conf_cruise_speed_set - self.conf_cruise_speed_set*(self.conf_forecast_dis - stStatic.state_reldis)/self.conf_forecast_dis
                elif tmp_state_step_static == 'Curve':
                    veh_speed_set_static = self.conf_cruise_speed_set - stStatic.state_param*self.conf_curve_speed_set_curvcoef + stStatic.state_reldis*self.conf_curve_speed_set_discoef
                else:
                    veh_speed_set_static = 0

                # vel_set_dynamic
                if tmp_state_step_dynamic == 'Cruise':
                    veh_speed_set_dynamic = self.conf_cruise_speed_set
                else:
                    veh_speed_set_dynamic = sorted((0, stDynamic.state_param , self.conf_cruise_speed_set))[1]

        Args:
            * stStatic: Static state information
            * stDynamic: Dynamic state information

        Returns:
            * veh_speed_set_filt: Vehicle velocity set point [m/s]

        """
        # Determination of velocity set from static state
        tmp_state_step_static = stStatic.state
        if tmp_state_step_static == 'Cruise':
            veh_speed_set_static = self.conf_cruise_speed_set
        elif tmp_state_step_static == 'Tl_stop':
            tmp_state_reldis_step = stStatic.state_reldis
            veh_speed_set_static = self.conf_cruise_speed_set - self.conf_cruise_speed_set*(self.conf_forecast_dis - tmp_state_reldis_step)/self.conf_forecast_dis
        elif tmp_state_step_static == 'Curve':
            tmp_param_step = stStatic.state_param
            tmp_reldis_step = stStatic.state_reldis
            # output saturation
            veh_speed_set_curve = self.conf_cruise_speed_set - tmp_param_step*self.conf_curve_speed_set_curvcoef + tmp_reldis_step*self.conf_curve_speed_set_discoef
            # veh_speed_set_static = sorted((self.conf_mincv_speed_set, veh_speed_set_curve, self.conf_cruise_speed_set))[1]
            veh_speed_set_static = veh_speed_set_curve
        else:
            veh_speed_set_static = 0
        # Determination of velocity set from dynamic state
        tmp_state_step_dynamic = stDynamic.state
        if tmp_state_step_dynamic == 'Cruise':
            veh_speed_set_dynamic = self.conf_cruise_speed_set
        else:
            tmp_preveh_vel = stDynamic.state_param # have to set the cruise speed set
            veh_speed_set_dynamic = sorted((0, tmp_preveh_vel, self.conf_cruise_speed_set))[1]

        veh_speed_set_dynamic_filt = Filt_LowPass(veh_speed_set_dynamic, self.veh_speed_set_dynamic, self.conf_filtnum_spdset,self.Ts_Loc)
        veh_speed_set_static_filt = Filt_LowPass(veh_speed_set_static, self.veh_speed_set_static, self.conf_filtnum_spdset,self.Ts_Loc)

        veh_speed_set = min(veh_speed_set_dynamic,veh_speed_set_static)
        veh_speed_set_filt = Filt_LowPass(veh_speed_set,self.veh_speed_set,self.conf_filtnum_spdset,self.Ts_Loc)

        self.veh_speed_set = veh_speed_set_filt
        self.veh_speed_set_dynamic = veh_speed_set_dynamic_filt
        self.veh_speed_set_static = veh_speed_set_static_filt
        return [veh_speed_set_filt, veh_speed_set_static_filt, veh_speed_set_dynamic_filt]

    def Lon_control(self,veh_vel_set, veh_vel):
        """Determine driver's acceleration and brake pedal position according to velocity set point

        Args:
            * veh_vel_set: Velocity set point [m/s]
            * veh_vel: Current vehicle velocity [m/s]

        Returns:
            * u_acc: Driver's acceleration pedal position [-]
            * u_brk: Driver's brake pedal position [-]
        """
        # State definition - Hysteresis filter with shift time
        vel_error = veh_vel_set - veh_vel

        if vel_error >= 0:
            control_mode = 1
        elif vel_error < -0.01:
            control_mode = 2
        else:
            control_mode = 0

        control_mode_filt = self.Filt_LonShiftTrnsDelay.filt_delay(control_mode)

        u_acc_ctl = self.Lon_Controller_acc.Control(veh_vel_set,veh_vel)
        u_brk_ctl = self.Lon_Controller_brk.Control(veh_vel,veh_vel_set)

        if control_mode == 1:
            u_acc = u_acc_ctl;
            u_acc_raw = sorted((0., u_acc, 1.))[1]
            u_acc_filt = Filt_LowPass(u_acc_raw, self.u_acc, self.conf_filtnum_pedal, self.Ts_Loc)
            u_brk_filt = 0.;
            self.Lon_Controller_brk.I_val_old = 0;
        elif control_mode == 2:
            u_brk = u_brk_ctl;
            u_brk_raw = sorted((0., u_brk, 1.))[1]
            u_brk_filt = Filt_LowPass(u_brk_raw, self.u_brk, self.conf_filtnum_pedal, self.Ts_Loc)
            u_acc_filt = 0.;
            self.Lon_Controller_acc.I_val_old = 0;
        else:
            u_brk_raw = 0
            u_acc_raw = 0
            u_acc_filt = Filt_LowPass(u_acc_raw, self.u_acc, self.conf_filtnum_pedal, self.Ts_Loc)
            u_brk_filt = Filt_LowPass(u_brk_raw, self.u_brk, self.conf_filtnum_pedal, self.Ts_Loc)
            self.Lon_Controller_brk.I_val_old = 0;
            self.Lon_Controller_acc.I_val_old = 0;
        # Set value
#        if stControl == 'acc':
#            acc_out_raw = trq_set/100
#            acc_out = Filt_LowPass(acc_out_raw, self.u_acc,self.conf_act_filt ,self.Ts_Loc)
#            brk_out = 0
#        elif stControl == 'brk':
#            acc_out = 0
#            brk_out_raw = -trq_set/100
#            brk_out = Filt_LowPass(brk_out_raw, self.u_brk,self.conf_act_filt ,self.Ts_Loc)
#        elif stControl == 'idle':
#            acc_out = 0
#            brk_out = 0
#        else:
#            acc_out = 0
#            brk_out = 0
#        self.trq_set_lon = trq_set

        self.stLonControl = control_mode_filt
        self.u_acc = u_acc_filt
        self.u_brk = u_brk_filt
        return [self.u_acc, self.u_brk]

    def Lon_behavior(self,static_obj_in, veh_position_s, road_len, veh_speed, pre_veh_speed = 'None', pre_veh_reldis = 250):
        """Simulate driver's longitudinal behaviors according to driving state

        Args:
            * static_obj_in: Static object information of driving route
            * road_len: Road length of driving route
            * veh_position_s: Current vehicle position on environment
            * veh_speed: Current vehicle velocity [m/s]
            * pre_veh_speed: Preceding vehicle velocity [m/s]
            * pre_veh_reldis: Relative distance to preceding vehicle [m]

        Returns:
            * acc_out: Acceleration pedal position [-]
            * brk_out: Brake pedal position [-]

        Include:
            * ``Mod_Behavior(Static_state_recog)`` : Determine static state
            * ``Mod_Behavior(Dynamic_state_recog)`` : Determine dynamic state
            * ``Mod_Behavior(Lon_vel_set)`` : Set velocity set point
            * ``Mod_Behavior(Lon_control)`` : Control driver action
        """
        stStatic = self.Static_state_recog(static_obj_in, veh_position_s, road_len)
        stDynamic = self.Dynamic_state_recog(pre_veh_speed, pre_veh_reldis)
        [veh_speed_set, veh_speed_set_static, veh_speed_set_dynamic] = self.Lon_vel_set(stStatic, stDynamic)
        [acc_out, brk_out] = self.Lon_control(veh_speed_set, veh_speed)
        return acc_out, brk_out

    def Lateral_state_recog(self, veh_position_x, veh_position_y, veh_ang, road_x, road_y):
        """Lateral state recognition according to road offset and heading angle

        Args:
            * veh_position_x: Vehicle position x on environment
            * veh_position_y: Vehicle position y on environment
            * veh_ang: Vehicle heading angle
            * road_x: Horizontal geometric information of environment
            * road_y: Vertical geometric information of environment

        Returns:
            * stLateral: Lateral state
                * angle_diff: Heading angle difference [rad]
                * lat_offset: Road offset [m]
        """
        stLateral = type_drvstate()
        [lon_offset, lat_offset, direction, min_index, veh_an, road_an] = Calc_PrDis(road_x, road_y, [veh_position_x, veh_position_y])
        if veh_ang < 0:
            veh_ang = veh_ang + 2*pi
        angle_diff = road_an - veh_ang
        if angle_diff >= pi/2:
            angle_diff = angle_diff - 2*pi
        elif angle_diff <= -pi/2:
            angle_diff = angle_diff + 2*pi
        else:
            angle_diff = angle_diff
        stLateral.set_state(direction, angle_diff, lat_offset)
        self.state_veh_an = veh_an
        self.road_an = road_an
        return stLateral

    def Lat_control(self,lane_offset, angle_diff, offset_des = 0, angle_diff_des = 0):
        """Determine driver's steering according to lateral offset, heading angle

        Args:
            * lat_offset: Road offset [m]
            * angle_diff: Heading angle difference [rad]
            * offset_des: Desired lateral offset (initial = 0) [m]
            * angle_diff_des: Desired angular offset (initial = 0) [rad]

        Returns:
            * steer_out_filt: Driver's steering value [-]
        """
        steer_out_offset = self.Lat_Controller_offset.Control(offset_des,lane_offset)
        steer_out_offset = sorted((-100., steer_out_offset, 100.))[1]
        #steer_out_offset_filt = Filt_LowPass(steer_out_offset,self.u_steer_offset,self.conf_filtnum_steer,self.Ts_Loc)

        steer_out_yaw = self.Lat_Controller_yaw.Control(angle_diff_des,-angle_diff)
        steer_out_yaw = sorted((-100., steer_out_yaw, 100.))[1]
        #steer_out_yaw_filt = Filt_LowPass(steer_out_yaw,self.u_steer_yaw,self.conf_filtnum_steer,self.Ts_Loc)

        steer_out = steer_out_offset + steer_out_yaw
        steer_out = sorted((-1., steer_out, 1.))[1]
        steer_out_filt = Filt_LowPass(steer_out,self.u_steer,self.conf_filtnum_steer,self.Ts_Loc)

        self.u_steer_offset = steer_out_offset
        self.u_steer_yaw = steer_out_yaw
        self.u_steer = steer_out_filt
        return steer_out_filt
    def Lat_behavior(self, veh_position_x, veh_position_y, veh_ang, road_x, road_y):
        """Simulate driver's lateral behaviors according to driving state

        Args:
            * veh_position_x: Vehicle position x on environment
            * veh_position_y: Vehicle position y on environment
            * veh_ang: Vehicle heading angle
            * road_x: Horizontal geometric information of environment
            * road_y: Vertical geometric information of environment

        Returns:
            * u_steer: Driver's steering value [-]

        Include:
            * ``Mod_Behavior(Lateral_state_recog)``: Determine static state
            * ``Mod_Behavior(Lat_control)``: Control driver action
        """
        self.stLateral = self.Lateral_state_recog(veh_position_x, veh_position_y, veh_ang, road_x, road_y)
        angle_offset = self.stLateral.state_param
        lane_offset = self.stLateral.state_reldis
        if self.stLateral.state == 'Left':
            lane_offset = -lane_offset
        else:
            lane_offset = lane_offset
        self.lane_offset = lane_offset
        self.psi_offset = angle_offset
        u_steer = self.Lat_control(lane_offset, angle_offset)
        return u_steer
#%%  ----- test ground -----
if __name__ == "__main__":
    pass
