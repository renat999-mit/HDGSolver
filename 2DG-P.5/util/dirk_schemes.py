#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 14:07:00 2023

@author: renatotronofigueras
"""

import numpy as np

def DIRK_3_3():
    
    gamma = 0.435866521508458
    
    tau2 = (gamma**2 - 3/2*gamma + 1/3)/(gamma**2 - 2*gamma + 1/2)
    
    b1 = (1/2*tau2 - 1/6)/((tau2 - gamma)*(1 - gamma))
    
    b2 = (1/2*gamma - 1/6)/((gamma - tau2)*(1 - tau2))
    
    A = np.array([[gamma,0.,0.],[tau2-gamma,gamma,0.],[b1,b2,gamma]])
    
    b = np.array([b1,b2,gamma])
    
    return A,b

def DIRK_2_2():
    
    gamma = 1/2 *(2 -np.sqrt(2))
    
    A = np.array([[gamma,0.],[1-gamma,gamma]])
    
    b = np.array([1-gamma,gamma])
    
    return A,b