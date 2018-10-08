# -*- coding: utf-8 -*-
"""
simulation models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Author
-------
* Kyunghan <kyunghah.min@gmail.com>
* Seungeon <bsewo5959@gmail.com>

Description
-----------
* Simulation models

Models
------------
* model_power: Power source of EV
    * modules: motor, battery, power
* model_vehicle: Vehicle models
    * modules: body, vehicle
* model_maneuver: Driver behavior and characteristics
    * modules: driver, behavior
* model_environment: Driving environment with road info
    * modules: env

Update
-------
* [18/05/08] - Initial release - kyunghan
* [18/10/08] - Model restructure - kyunghan

"""

import pysim.models.model_power
import pysim.models.model_vehicle
import pysim.models.model_maneuver
import pysim.models.model_environment
