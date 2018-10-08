
# import package modules
"""
Simulation model : Power
==============================================================

Author
~~~~~~~~~~~~~
* kyunghan <kyunghah.min@gmail.com>

Description
~~~~~~~~~~~~~
* Simulate power source of EV

Modules detail
~~~~~~~~~~~~~
* Motor module - calculate mechanical loss and efficiency of motor
    * aa
* Battery module - calculate battery power consumption

Update
~~~~~~~~~~~~~
* [18/05/31] - Initial release - kyunghan
* [18/06/01] - Add battery class - seungeon
* [18/06/01] - Modulization - kyuhwan
  - Powertrain class -> Divide into motor, battery classes
  - Powertrain class has composition relationship with motor, battery
* [18/06/11] - Modification - kyunghan
  - Modify drivetrain module to body module
  - Modify model configurations
* [18/10/08] - Modification - kyunghan
  - Move the motor_driven function to power module
  - Battery module restructure
"""
# simulation sampling time
Ts = 0.01
"""global vairable: simulation sampling timeself.

you can declare other sampling time in application as vairable ``Ts``

"""
import numpy as np
from pysim.sub_util.sub_type_def import type_pid_controller
#%% Motor class
class Mod_Motor:
    def __init__(self):
        self.Motor_init_config()
        self.Motor_config()
        self.Ts_loc = Ts

    def Motor_config(self, loss_mech_C0 = 0.0206, loss_mech_C1 = -2.135e-5, loss_copper_C0 = 0.2034, loss_stray_C0 = 3.352e-6, loss_stray_C1 = -2.612e-9, loss_iron_C0 = 1e-6, loss_iron_C1 = 1.55):
        """Motor parameter configuration

        Parameters not specified are declared as default values

        If you want set a specific parameter don't use this function,
        just type::

            >>> Mod_PowerTrain.conf_rm_mot = 0.2
            ...

        Args:
            Motor parameter values, default values are setted
        """
        self.loss_mech_C0 = loss_mech_C0
        self.loss_mech_C1 = loss_mech_C1
        self.loss_copper_C0 = loss_copper_C0
        self.loss_stray_C0 = loss_stray_C0
        self.loss_stray_C1 = loss_stray_C1
        self.loss_iron_C0 = loss_iron_C0
        self.loss_iron_C1 = loss_iron_C1

    def Motor_init_config(self):
        """Initialize motor state

        Initialize motor state

        Args:
            Motor voltage and current
            Motor rotational speed
            Motor torque
            Motor power loss
        """
        self.v_mot = 320
        self.i_mot = 0
        self.w_mot = 0
        self.t_mot = 0
        self.p_mot_mech = 0
        self.p_mot_loss = 0
        self.p_mot_elec = 0

    def Motor_Power_system(self, t_mot, w_mot):
        """Initialize motor state

        Initialize motor state

        Args:
            Motor voltage and current
            Motor rotational speed
            Motor torque
            Motor power loss
        """
        self.p_mot_mech = t_mot * w_mot
        w_mot = sorted([0,w_mot,10000])[1]
        self.p_mot_loss = (self.loss_mech_C0 + self.loss_mech_C1*w_mot)*w_mot**2 + self.loss_copper_C0*t_mot + (self.loss_stray_C0+self.loss_stray_C1*w_mot)*w_mot**2*t_mot**2 + self.loss_iron_C0*w_mot**self.loss_iron_C1*t_mot**2
        self.p_mot_elec = self.p_mot_mech + self.p_mot_loss
        self.Motor_Elec_system(self.p_mot_elec)
        return self.p_mot_elec

    def Motor_Elec_system(self, p_mot_elec):
        self.i_mot = p_mot_elec / self.v_mot
        return self.i_mot

#%% Battery class
class Mod_Battery:
    def __init__(self):
        # Additional power consumption [0.1 W]
        self.Battery_config()
        self.Battery_init_config()


    def Battery_config(self, Voc_base_voltage = 356.0, E_tot = 230400000.0, MaxPower = 150000.0,
                       R0 = 0.0016, R1_a=76.5218, R1_b=-7.9563, R1_c=23.8375,
                       C1_a = -649.8350, C1_b = -64.2924, C1_c = 12692.1946, R2_a = 5.2092, R2_b = -35.2367, R2_c = 124.9467, C2_a = -78409.2788,
                       C2_b = -0.0131, C2_c = 30802.62582, Int_E = 161280000.0):

        self.conf_Voc_base = Voc_base_voltage
        self.conf_Etot = E_tot
        self.conf_MaxPower = MaxPower

        self.conf_R0 = R0
        self.conf_R1_a = R1_a;        self.conf_R1_b = R1_b;        self.conf_R1_c = R1_c
        self.conf_C1_a = C1_a;        self.conf_C1_b = C1_b;        self.conf_C1_c = C1_c
        self.conf_R2_a = R2_a;        self.conf_R2_b = R2_b;        self.conf_R2_c = R2_c;
        self.conf_C2_a = C2_a;        self.conf_C2_b = C2_b;        self.conf_C2_c = C2_c;

        self.soc_data_value = [0, 6.25,  31.25,  62.5,   93.75, 100.00]
        self.soc_data_index_Voc_rate = [0, 0.625, 0.8125,  0.875,  1.00,   1.0915]

    def Battery_init_config(self, SOC_init = 70, power_additional_set = 500):
        self.power_additional = power_additional_set # [0.1 W]
        self.SOC = SOC_init
        self.Voc = self.conf_Voc_base
        self.Current = 0
        self.Vterm = 0
        self.V1 = 0
        self.V2 = 0
        self.Internal_Energy = self.conf_Etot * SOC_init/100

    def Calc_Voc(self, input):
        self.Voc_rate = np.interp(input, self.soc_data_value, self.soc_data_index_Voc_rate)
        self.Voc = self.conf_Voc_base * self.Voc_rate
        return self.Voc

    def Calc_SOC(self, current):
        self.V1 = self.V1 + (current/self.C1 - self.V1/(self.R1*self.C1) )*Ts
        self.V2 = self.V2 + (current/self.C2 - self.V2/(self.R2*self.C2) )*Ts
        self.Internal_Energy = self.Internal_Energy - self.Voc * current * Ts
        self.SOC = self.Internal_Energy / self.conf_Etot * 100
        return self.SOC

    def Calc_Vterm(self, ):
        self.Vterm = self.Voc - self.Current * self.conf_R0 - self.V1 - self.V2
        return self.Vterm

    def Calc_Current(self, Motor_Net_Power):
        Voc = self.Calc_Voc(self.SOC)
        self.R1 = self.conf_R1_a * np.exp(self.conf_R1_b * self.SOC) + self.conf_R1_c
        self.C1 = self.conf_C1_a * np.exp(self.conf_C1_b * self.SOC) + self.conf_C1_c
        self.R2 = self.conf_R2_a * np.exp(self.conf_R2_b * self.SOC) + self.conf_R2_c
        self.C2 = self.conf_C2_a * np.exp(self.conf_C2_b * self.SOC) + self.conf_C2_c

        self.Consume_power = self.power_additional + Motor_Net_Power
        self.Current = (Voc - np.sqrt(Voc**2 - 4*self.conf_R0 * self.Consume_power)) / (2 * self.conf_R0)

        self.Calc_SOC(self.Current)
        return self.Current

#%% Powertrain class
class Mod_Power():
    def __init__(self, Mod_Motor_loc = Mod_Motor(), Mod_Battery_loc = Mod_Battery()):
        self.ModMotor = Mod_Motor_loc
        self.ModBattery = Mod_Battery_loc

    def Motor_driven(self, torque_set = 0, w_drivtrain = 0):
        # Elec motor model: Motor torque --> Mech motor model: Motor speed --> Drive shaft model: Load torque
        """Motor driven function

        Generate motor output(torque, speed) and load torque according to input voltage and wheel speed (shaft speed = wheel speed)

        Contain theree modules ``Elecic dynamics``, ``Menahicla dynamics``, ``Shaft dynamics``

        Args:
            * v_in: motor input voltage [V]
            * w_shaft: rotational speed of drive shaft from body model [rad/s]

        returns:
            * t_mot: motor torque [Nm]
            * w_mot: motor rotational speed [rad/s]
            * t_load: load torque from body model [Nm]
        """
        self.t_mot = torque_set
        self.w_mot = w_drivtrain
        p_mot_elec = self.ModMotor.Motor_Power_system(self.t_mot,self.w_mot)
        self.ModBattery.Calc_Current(p_mot_elec)
        return self.w_mot, self.t_mot
