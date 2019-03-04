# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:56:13 2018

@author: Kyunghan
"""

import numpy as np
from keras import backend as K


class test_fcn:
    
    def __init__(self):
        self.calc_fcn = self.build_function()
        
    def build_function(self):         
        calc_fcn = lambda a,b: a*b
        return calc_fcn
        

if __name__ == "__main__":
    test_calc_fcn = test_fcn()    
    
    
    