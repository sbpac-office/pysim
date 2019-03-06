# -*- coding: utf-8 -*-
"""
Simulation model : Vehicle
==============================================================

Description
~~~~~~~~~~~~~
* Simulate vehicle dynamics of EV

Modules summary
~~~~~~~~~~~~~~~~~
* Body module - simulate dynamics of drivetrain
    * Drivetrain_config - configure drive train Parameters
    * Lon_equivalence - calculate rotational dynamics of equivalent vehicle
    * Driveshaft_dynamics - calculate rotational dynamics of drive shaft
    * Tire_dynamics - calculate rotational dynamics of tire
    * Motor_dynamics - calculate rotational dynamics of motor out

* Vehicle module - contain ``power module`` and ``body module``, simulate vehicle behavior
    * Veh_init_config - configure initial vehicle state, position, velocity
    * Veh_config - configure vehicle parameters
    * Veh_position_update - update vehicle position according to speed and steering
    * Veh_driven - calculate vehicle velocity and wheel theta
        * Veh_lon_driven - simulate longitudinal vehicle behavior (include ``Mod_Power(Motor_driven)``, ``Mod_Body(Lon_equivalence)``)
            * Veh_lon_dynamics - calculate vehicle acceleration according to traction force and drag force
            * Acc_system - determine desired torque set according to acceleration pedal position
            * Brake_system - determine brake torque set according to brake pedal position
            * Drag_system - determine air and rolling resistance force
        * Veh_lat_driven - simulate lateral vehicle behavior
            * Veh_lat_dynamics - calculate tire wheel dynamics according to steering input


* Module diagram::

    Veh_driven( Veh_lon_driven, Veh_lat_driven )
        >> Veh_lon_driven( Veh_lon_dynamics, Acc_system, Brake_system, Drag_system, Motor_driven, Lon_equivalence )
            >> Lon_equivalence in Mod_Body
            >> Motor_driven in Mod_Power
        >> Veh_lat_driven( Veh_lat_dynamics )

Update
~~~~~~~~~~~~~
* [18/05/31] - Initial release - Kyunghan
* [18/06/01] - Seperate powertrain class in new file - Kyuhwan
* [18/06/11] - Modification - Kyunghan
  - Modify drivetrain module to body module
  - Modify model configurations
  - Add regeneration module in body.brake_system
* [18/08/08] - Restructure - Kyunghan
  - Modify drive train module
  - Modify tire module
  - Modify drag force module
"""

# import python lib modules
from math import pi, sin, cos, atan
import numpy as np
from scipy.spatial import distance as dist_calc
# import package modules
from pysim.sub_util.sub_utilities import Calc_Radius, Filt_MovAvg, Calc_PrDis
from pysim.sub_util.sub_type_def import type_pid_controller, type_drvstate, type_objective
# import config data modules

# simulation sampling time
#Ts = 0.01
"""global vairable: simulation sampling timeself.

you can declare other sampling time in application as vairable ``Ts``

"""


class Mod_Body:
    """
    * Body module
    """
    def __init__(self, Ts = 0.01):
        self.t_mot_load = 0
        self.t_driven = 0
        self.t_wheel_load = 0
        self.t_wheel_traction_f = 0
        self.w_wheel = 0
        self.w_shaft = 0
        self.w_motor = 0
        self.w_vehicle = 0
        self.Drivetrain_config()        
        self.Ts_loc = Ts

    def Drivetrain_config(self, conf_rd_wheel = 0.301, conf_jw_wheel = 0.1431, conf_jw_diff_in = 0.015, conf_jw_diff_out = 0.015, conf_jw_trns_out = 0.015, conf_jw_trns_in = 0.01, conf_jw_mot = 0.005,
                    conf_eff_trns = 0.96, conf_eff_diff = 0.9796, conf_eff_diff_neg = 0.9587, conf_gear = 6.058, conf_mass_veh = 1200, conf_mass_add = 0):
        """Drivetrain and body parameter configuration
        Parameters not specified are declared as default values
        If you want set a specific parameter don't use this function,
        just type::

            Mod_Body.conf_veh_len = 2
            ...

        Parameters:
            Vehicle mass
            Wheel radius
            Momentum of inertia: shaft, wheel
            Drive shaft efficiency, gear ratio
        """
        self.conf_gear = conf_gear
        self.conf_mass_veh = conf_mass_veh + conf_mass_add
        self.conf_rd_wheel = conf_rd_wheel
        # rotational inertia
        self.conf_jw_wheel = conf_jw_wheel
        self.conf_jw_diff_in = conf_jw_diff_in
        self.conf_jw_diff_out = conf_jw_diff_out
        self.conf_jw_trns_in = conf_jw_trns_in
        self.conf_jw_trns_out = conf_jw_trns_out
        self.conf_jw_mot = conf_jw_mot
        # equivalence inertia
        self.conf_jw_wheel_eq_f = conf_jw_wheel + conf_jw_diff_out*2
        self.conf_jw_wheel_eq_r = conf_jw_wheel + conf_jw_diff_out
        self.conf_jw_shaft_eq = (conf_jw_mot*conf_gear + conf_jw_trns_in*conf_gear + conf_jw_trns_out + conf_jw_diff_in + conf_jw_diff_out + conf_jw_wheel)*2;
        self.conf_jw_body_eq = self.conf_mass_veh * conf_rd_wheel**2 + 2*(self.conf_jw_wheel_eq_f+self.conf_jw_wheel_eq_r)
        self.conf_jw_vehicle_eq = self.conf_jw_shaft_eq + self.conf_jw_body_eq
        self.conf_jw_vehicle = self.conf_mass_veh * conf_rd_wheel**2
        # efficiency
        self.conf_eff_trns = conf_eff_trns
        self.conf_eff_diff = conf_eff_diff
        self.conf_eff_diff_neg = conf_eff_diff_neg
        # equivalence efficiency
        self.conf_eff_eq_pos = 1 - (1-conf_eff_trns + conf_eff_trns*(1-conf_eff_diff))
        self.conf_eff_eq_neg = 1 - ((1/conf_eff_trns - 1) + conf_eff_trns*(1/conf_eff_diff_neg - 1))


    def Lon_equivalence(self,t_mot, t_brk, t_drag):
        """Calculate equivalent rotational dynamics of vhielce

        Equivalet component: Motor + Shaft + Wheel + Vehicle

        Dynamics::

            w_vehicle_dot = (t_motor - t_drag - t_brake) / j_veh
            ...

        Args:
            * t_mot: motor torque [Nm]
            * t_brk: brake torque [Nm]
            * t_drag: equivalent drag torque [Nm]
        Return:
            * load torque: load torque of each component [Nm]
            * f_lon: longitudinal traction force [N]
        """
        # shaft torque calculation
        t_motor_in = t_mot * self.conf_gear
        t_brk_wheel = t_brk/4
        if t_motor_in >= 0:
            t_shaft_loss = t_motor_in*(1-self.conf_eff_eq_pos)
        else:
            t_shaft_loss = t_motor_in*(1-self.conf_eff_eq_neg)
        t_shaft_in = t_motor_in - t_shaft_loss
        # vehicle equivalence
        t_driven = t_shaft_in - t_brk - t_drag
        w_dot_wheel = t_driven/self.conf_jw_vehicle_eq
        self.w_vehicle = self.w_vehicle + w_dot_wheel*self.Ts_loc
        if self.w_vehicle <= -0.01:
            w_dot_wheel = 0
            self.w_vehicle = 0
            t_brk_wheel = 0
        # load torque calculation - vehicle traction
        t_veh_traction = self.conf_jw_vehicle * w_dot_wheel + t_drag
        f_lon = t_veh_traction/self.conf_rd_wheel
        # load torque calculation
        t_shaft_out = t_shaft_in - self.conf_jw_shaft_eq*w_dot_wheel
        # load torque calculation
        t_wheel_in = t_shaft_out/2
        # load torque calculation - wheel load
        t_wheel_traction_r = -t_brk_wheel-w_dot_wheel*self.conf_jw_wheel_eq_r
        t_wheel_traction_f = (t_veh_traction - 2*t_wheel_traction_r )/2 + t_brk_wheel
        # load torque calculation - motor load
        t_mot_load = t_mot - w_dot_wheel*self.conf_gear*(self.conf_jw_mot + self.conf_jw_trns_in)
        self.t_mot_load = t_mot_load
        self.t_driven = t_driven
        self.t_shaft_in = t_shaft_in
        self.t_shaft_out = t_shaft_out
        self.t_shaft_loss = t_shaft_loss
        self.t_wheel_in = t_wheel_in
        self.t_wheel_traction_f = t_wheel_traction_f
        self.w_dot_vehicle = w_dot_wheel
        return t_mot_load, t_shaft_in, t_shaft_out, t_wheel_in, t_wheel_traction_f, t_driven, f_lon

    def Driveshaft_dynamics(self, t_shaft_in, t_shaft_out, w_shaft):
        """Calculate rotational dynamics of shaft

        Args:
            * t_shaft_in: shaft in torque = motor_out torque - mechanical loss [Nm]
            * t_shaft_out: shaft out torque [Nm]
            * w_shaft: previous rotational speed of shaft [rad/s]
        Return:
            * w_shaft: updated rotational speed of shaft [rad/s]
        """
        w_dot_shaft = (t_shaft_in - t_shaft_out)/self.conf_jw_shaft_eq
        w_shaft = w_shaft + self.Ts_loc * w_dot_shaft
        self.w_dot_shaft = w_dot_shaft
        self.w_shaft = w_shaft
        return w_shaft

    def Tire_dynamics(self, t_wheel_load, t_wheel_traction_f, t_brk, w_wheel):
        """Calculate rotational dynamics of wheel

        Args:
            * t_wheel_load: wheel in torque = shaft out torque [Nm]
            * t_wheel_traction_f: wheel out torque to traction force [Nm]
            * t_brk: brake torque [Nm]
            * w_shaft: previous rotational speed of wheel [rad/s]
        Return:
            * w_shaft: updated rotational speed of wheel [rad/s]
        """
        t_brk_w = t_brk/4
        t_brk_w = 0
        w_dot_wheel = (t_wheel_load - t_wheel_traction_f - t_brk_w)/self.conf_jw_wheel_eq_f
        #if w_wheel <=0:
        #    w_dot_wheel = 0
        w_wheel = w_wheel + self.Ts_loc*w_dot_wheel
        self.w_dot_wheel = w_dot_wheel
        self.w_wheel = w_wheel
        return w_wheel

    def Motor_dynamics(self, t_mot, t_mot_load, w_motor):
        """Calculate rotational dynamics of motor

        Args:
            * t_mot: motor generated torque [Nm]
            * t_mot_load: motor load torque to shaft [Nm]
            * w_shaft: previous rotational speed of motor [rad/s]
        Return:
            * w_shaft: updated rotational speed of motor [rad/s]
        """
        w_dot_motor = (t_mot - t_mot_load)/(self.conf_jw_mot + self.conf_jw_trns_in)
        w_motor = w_motor + self.Ts_loc*w_dot_motor
        self.w_dot_motor = w_dot_motor
        self.w_motor = w_motor
        return w_motor


class Mod_Veh:
    """

    * Vehicle module: Set the ``power`` and ``body`` model when initialization

    """
    def __init__(self,powertrain_model,drivetrain_model,Ts = 0.01):
        self.ModPower = powertrain_model
        self.ModDrive = drivetrain_model
        self.Ts_loc = Ts
        self.Veh_init_config()
        self.Veh_config()        

    def Veh_init_config(self, x_veh = 0, y_veh = 0, s_veh = 0, n_veh = 0, psi_veh = 0, vel_veh = 0, theta_wheel = 0):
        """Initialize vehicle state

        State:
            * Vehicle position on environment
            * Velocity
            * Heading angle
        """
        self.pos_x_veh = x_veh
        self.pos_y_veh = y_veh
        self.pos_s_veh = s_veh
        self.pos_n_veh = n_veh
        self.psi_veh = psi_veh
        self.vel_veh = vel_veh
        self.the_wheel = theta_wheel
        self.veh_acc = 0
        self.t_mot_reg_set = 0
        self.t_brake = 0
        self.t_mot = 0

    def Veh_config(self, conf_drag_air_coef = 0, conf_add_weight = 0, conf_drag_ca = 143.06, conf_drag_cc = 0.4405,
                   conf_veh_len = 2,conf_acc_trq_fac = 82.76, conf_brk_trq_fac = 501.8, conf_motreg_max = 100):
        """Vehicle parameter configuration
        Parameters not specified are declared as default values
        If you want set a specific parameter don't use this function,
        just type::

            >>> Mod_Body.conf_veh_len = 2
            ...

        Parameters:
            * Vehicle size
            * Air drag coefficient
            * Rolling resistance coefficient
            * Acceleration, Brake coefficient
        """
        self.conf_drag_air_coef = conf_drag_air_coef
        self.conf_drag_ca = conf_drag_ca
        self.conf_drag_cc = conf_drag_cc
        self.conf_brk_trq_fac = conf_brk_trq_fac
        self.conf_acc_trq_fac = conf_acc_trq_fac
        self.conf_veh_len = conf_veh_len
        self.conf_veh_mass = self.ModDrive.conf_mass_veh
        self.conf_rd_wheel = self.ModDrive.conf_rd_wheel
        self.conf_motreg_max = conf_motreg_max
        self.swtRegCtl = 0

    def Veh_position_update(self, vel_veh = 0, the_wheel = 0):
        """Update vehicle position on environment

        Args:
            * vel_veh: Vehicle velocity [m/s]
            * the_wheel: Wheel angle [rad]
        Return:
            * x, y position: Absolute vehicle coordinate [m]
            * s, n position: Road relative vehicle position [m]
            * psi_veh: Vehicle heading angle [rad]
        """
        veh_len = self.conf_veh_len
        ang_veh = the_wheel + self.psi_veh
        x_dot = vel_veh*cos(ang_veh)
        self.pos_x_veh = self.pos_x_veh + x_dot*self.Ts_loc
        y_dot = vel_veh*sin(ang_veh)
        self.pos_y_veh = self.pos_y_veh + y_dot*self.Ts_loc
        s_dot = vel_veh*cos(the_wheel)
        self.pos_s_veh = self.pos_s_veh + s_dot*self.Ts_loc
        n_dot = vel_veh*sin(the_wheel)
        self.pos_n_veh = self.pos_n_veh + n_dot*self.Ts_loc
        psi_dot = vel_veh/veh_len*the_wheel
        self.psi_veh = self.psi_veh + psi_dot*self.Ts_loc
        return [self.pos_x_veh, self.pos_y_veh, self.pos_s_veh, self.pos_n_veh, self.psi_veh]

    def Veh_driven(self, u_acc, u_brake, u_steer = 0):
        """Simulate vehicle behavior according to driver's input

        Args:
            * u_acc: Acceleration pedal position [-]
            * u_brake: Brake pedal position [-]
            * u_steer: Steering wheel angle [-]
        Return:
            * vel_veh: Vehicle velocity [m/s]
            * the_wheel: Wheel heading angle [rad]
        Include:
            * ``Mod_Veh(Veh_lon_driven,Veh_lat_driven)``: Simulate vehicle behavior

        """
        # Longitudinal driven
        vel_veh = self.Veh_lon_driven(u_acc, u_brake)
        # Lateral driven
        the_wheel = self.Veh_lat_driven(u_steer)
        return vel_veh, the_wheel

    def Veh_lon_driven(self, u_acc, u_brake):
        """Simulate longitudinal vehicle behavior according to driver's input

        Calculate total drag force from Drag_system in vehicle module

        Determine driven force from Acc_system and Brake_system in vehicle module

        Drive Lon_equivalence function in body module

        Args:
            * u_acc: Acceleration pedal position [-]
            * u_brake: Brake pedal position [-]
        Return:
            * veh_vel: Vehicle velocity [m/s]
            * s, n position: Road relative vehicle position [m]
            * psi_veh: Vehicle heading angle [rad]
        Include:
            * ``Mod_Veh(Acc_system, Brake_system, Drag_system, Veh_lon_dynamics)``: Determine driven torque
            * ``Mod_Body(Lon_equivalence)``: Calculate vehicle rotational dynamics
            * ``Mod_Power(Motor_driven)``: Determine motor driven torque

        """
        w_mot = self.ModDrive.w_motor
        w_shaft = self.ModDrive.w_shaft
        w_wheel = self.ModDrive.w_wheel
        # Calculation of torque set
        t_mot_set = self.Acc_system(u_acc)
        t_brk, t_mot_reg_set = self.Brake_system(u_brake, self.t_mot_reg_set)
        t_drag, f_drag = self.Drag_system(self.vel_veh)
        # Power control
        t_mot_des = t_mot_set - t_mot_reg_set
        w_mot, t_mot = self.ModPower.Motor_driven(t_mot_des, w_mot)
        self.t_mot_des = t_mot_des
        # Body equivalence
        t_mot_load, t_shaft_in, t_shaft_out, t_wheel_in, t_wheel_traction_f, t_driven, f_lon = self.ModDrive.Lon_equivalence(t_mot,t_brk,t_drag)
        self.f_lon = f_lon
        # Shaft dynamics
        w_mot = self.ModDrive.Motor_dynamics(t_mot, t_mot_load, w_mot)
        w_shaft = self.ModDrive.Driveshaft_dynamics(t_shaft_in, t_shaft_out, w_shaft)
        w_wheel = self.ModDrive.Tire_dynamics(t_wheel_in, t_wheel_traction_f, t_brk, w_wheel)
        # Vehicle dynamics
        self.vel_veh, self.veh_acc = self.Veh_lon_dynamics(f_lon, f_drag, self.vel_veh)
        return self.vel_veh

    def Veh_lon_dynamics(self, f_lon, f_drag, vel_veh):
        """Calculate longitudinal vehicle dynamics

        Equivalet component: Motor + Shaft + Wheel + Vehicle

        Dynamics::

            veh_acc = (f_lon - f_drag) / m_veh
            ...

        Args:
            * f_lon: longitudinal driven force from Lon_equivalence [N]
            * f_drag: longitudinal drag force from Drag_system, Brake_system [N]
        Return:
            * veh_vel: vehicle velocity [m/s]
        """
        veh_acc = (f_lon - f_drag)/self.conf_veh_mass
        vel_veh_calc = vel_veh + self.Ts_loc*veh_acc
        vel_veh = sorted((0., vel_veh_calc, 1000.))[1]
        return vel_veh, veh_acc

    def Veh_lat_driven(self, u_steer):
        """Simulate lateral vehicle behavior according to driver's input

        Args:
            * u_steer: steering input [-]
        Return:
            * the_wheel: wheel heading angle [rad]
        Include:
            * ``Mod_Veh(Veh_lat_dynamics)``: Calculate wheel dynamics
        """
        self.the_wheel = self.Veh_lat_dynamics(self.the_wheel, u_steer)
        return self.the_wheel

    def Veh_lat_dynamics(self, the_wheel, u_steer):
        """Calculate lateral vehicle dynamics

        Dynamics::

            the_wheel_dot = (u_steer - the_wheel) / lat_dynamic_coeff
            ...

        Args:
            * u_steer: steering input of driver [-]
            * the_wheel: longitudinal drag force from Drag_system, Brake_system [N]
        Return:
            * the_wheel: vehicle velocity [m/s]
        """
        the_wheel = the_wheel + self.Ts_loc/0.2*(u_steer - the_wheel)
        return the_wheel

    def Acc_system(self, u_acc):
        """Determine desired motor torque

        Args:
            * u_acc: acceleration pedal of driver [-]
        Return:
            * t_mot_des: desired motor torque [Nm]
        """
        self.t_mot_des = u_acc * self.conf_acc_trq_fac
        return self.t_mot_des

    def Brake_system(self, u_brake, t_reg_set = 0):
        """Determine brake torque

        Regen control::

            swtRegCtl = 0 # Meachanical braking
            swtRegCtl = 1 # Transfer braking torque to regenerative motor torque
            swtRegCtl = 2 # Set the specific regenerative motor torque

        Args:
            * u_brake: acceleration pedal of driver [-]
            * t_reg_set: Regenerative torque set [-]
        Return:
            * t_brake: Braking torque [Nm]
            * t_mot_reg: Regenerative motor torque [Nm]
        """

        t_brake_set = u_brake * self.conf_brk_trq_fac
        t_brake_set_mot_eq = t_brake_set/self.ModDrive.conf_gear

        # Regeneration control
        if self.swtRegCtl == 1:
        # Maximum regen
            if (self.conf_motreg_max - t_brake_set_mot_eq) >= 0:
                t_mot_reg = t_brake_set_mot_eq
                t_brake = 0
            else:
                t_mot_reg = self.conf_motreg_max
                t_brake = (t_brake_set - self.conf_motreg_max*self.ModDrive.conf_gear)
        elif self.swtRegCtl == 2:
        # Additional regen torque
            t_brake = t_brake_set
            t_mot_reg = t_reg_set
        else:
        # Meachanical braking
            t_brake = t_brake_set
            t_mot_reg = 0

        self.t_brake_set = t_brake_set
        self.t_brake = t_brake
        self.t_mot_reg = t_mot_reg
        return t_brake, t_mot_reg

    def Drag_system(self, vel_veh):
        """Calculate drag torque

        Air drag::

            f_drag_air = 0.5*1.25*self.conf_drag_air_coef*vel_veh**2

        Rolling resistance::

            f_drag_roll = self.conf_drag_ca + self.conf_drag_cc*vel_veh**2

        Args:
            * vel_veh: vehicle velocity [m/s]
        Return:
            * t_drag: total drag torque [Nm]
            * f_drag: total drag force [N]
        """
        f_drag_roll = self.conf_drag_ca + self.conf_drag_cc*vel_veh**2
        f_drag_air = 0.5*1.25*self.conf_drag_air_coef*vel_veh**2
        f_drag = f_drag_roll + f_drag_air
        if vel_veh < 0.1:
            f_drag = 0
        t_drag = f_drag * self.conf_rd_wheel
        self.f_drag = f_drag
        self.t_drag = t_drag
        return t_drag, f_drag
#%%  ----- test ground -----
if __name__ == "__main__":
    pass
