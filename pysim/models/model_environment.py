# -*- coding: utf-8 -*-
"""
Simulation model : Environment
==============================================================

Description
~~~~~~~~~~~~~
* Import road information for driving environment

Modules summary
~~~~~~~~~~~~~~~~~
* Env module - environment module
    * Env_config - configure environment
    * Road_static_object - determine road static objects
        * Road_curve_def - set parameters
    * Obj_add - add static object (Tl, Curve)
    * Veh_position_init - Set initial position of vehicle on the road, include (``Mod_Veh``)

Update
~~~~~~~~~~~~~
* [18/05/31] - Initial release - Kyunghan
* [18/06/05] - Modification of lon control - Kyunghan
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
Ts = 0.01
"""global vairable: simulation sampling timeself.

you can declare other sampling time in application as vairable ``Ts``

"""
class Mod_Env:
    """
    * Environment module: include road information (``road_x``, ``road_y``)
    """
    def __init__(self, road_array_x_in, road_array_y_in, start_road_len = 0):
        self.road_x = road_array_x_in
        self.road_y = road_array_y_in
        self.Env_config(road_array_x_in)
        self.Road_static_object(start_road_len)


    def Env_config(self, road_array_x_in, conf_mincurv_value = 0.001):
        self.object_list = [type_objective() for _ in range(len(road_array_x_in))]
        self.conf_mincurv_val = conf_mincurv_value


    def Road_static_object(self, start_road_len = 0):
        """Determine road static objects

        Arrange road length and road curvature according to global road information (``road_x``, ``road_y``)

        Include:

            * ``Mod_Env(Road_curve_def)``: Determine road curve position and curvature value

        """
        road_array_x_in = self.road_x
        road_array_y_in = self.road_y
        loc_env_road_s = np.zeros(len(road_array_x_in))
        loc_env_road_s[0] = start_road_len
        loc_env_road_ang = np.zeros(len(road_array_x_in))
        loc_env_road_ang[0] = 0
        for i in range(1,len(road_array_x_in),1):
            old_pos = [road_array_x_in[i-1],road_array_y_in[i-1]]
            new_pos = [road_array_x_in[i],road_array_y_in[i]]
            loc_env_road_s[i] = loc_env_road_s[i-1] + dist_calc.euclidean(old_pos, new_pos)
            loc_env_road_ang[i] = np.arctan((road_array_y_in[i] - road_array_y_in[i-1])/(road_array_x_in[i] - road_array_x_in[i-1]))
        self.road_ang = loc_env_road_ang
        self.road_len = loc_env_road_s
        self.object_list = self.Road_curve_def(road_array_x_in, road_array_y_in, loc_env_road_s)

    def Obj_add (self, object_in, object_param_in, object_s_location):
        """Add one object on the road

        Args:
            * object_in: Set the object class ('Tl', 'Curve', ..)
            * object_param_in: Set the object parameter ('Curvature', 'State')
            * object_s_location: Set the object location on the road
        """
        loc_env_road_s = self.road_len
        tmp_s_index = np.min(np.where(loc_env_road_s >= object_s_location)) - 1
        self.object_list[tmp_s_index].add_object(object_in,object_param_in,object_s_location)

    def Road_curve_def(self, road_array_x_in, road_array_y_in, loc_env_road_s):
        """Determine curve information from road data

        Calculate cuvature using road data then add curve object to static object list

        Args:
            * road_array_x_in: Horizontal geometric information of road
            * road_array_y_in: Vertical geometric information of road
            * loc_env_road_s: Road length information

        Returns:
            * object_list: Road object list for curve information
        """
        object_list = [type_objective() for _ in range(len(road_array_x_in))]
        [R_out, x_c_out, y_c_out, circle_index, mr_o, mt_o] = Calc_Radius(road_array_x_in, road_array_y_in, 3)
        tmp_Curve = 1/R_out
        tmp_Curve_Filt = Filt_MovAvg(tmp_Curve,3)
        tmp_Curve_index = np.arange(len(road_array_x_in))[tmp_Curve_Filt >= self.conf_mincurv_val]
        self.road_curve = tmp_Curve
        for i in range(len(tmp_Curve_index)):
            tmp_s_index = tmp_Curve_index[i]
            object_list[tmp_s_index].add_object('Curve',tmp_Curve[tmp_s_index],loc_env_road_s[tmp_s_index])
        return object_list

    def Vehicle_init_config(self, veh_mod, road_index = 0):
        """Set initial position and heading angle of vehicle on road model

        Include ``Mod_Veh`` when initialize

        Args:
            * veh_mod: Vehicle module
            * road_index: Road length index for vehicle position

        """
        veh_mod.pos_x_veh = self.road_x[road_index]
        veh_mod.pos_y_veh = self.road_y[road_index]
        veh_mod.psi_veh = atan((self.road_y[road_index+1] - self.road_y[road_index])/(self.road_x[road_index+1] - self.road_x[road_index])) + (1 - (self.road_x[road_index+1] - self.road_x[road_index])/abs((self.road_x[road_index+1] - self.road_x[road_index])))/2*pi
