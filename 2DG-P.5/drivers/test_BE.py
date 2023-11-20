#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:17:39 2023

@author: renatotronofigueras
"""

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt

from mesh import *
from mesh.mkmesh_1Dinterval import mkmesh_1Dinterval
from util import *
from master import *
from hdgker import hdg_solve, hdg_postprocess, DIRK_step, BE_step, newton_method

def exact_u(x, t=1, eps = 1):
    
    ul = 2
    ur = 0
    
    c = 1/2 * (ur+ul)
    
    return 1 - np.tanh((x - c*t)/(2*eps))

def exact_q(x,t = 1,eps = 1):
    
    ul = 2
    ur = 0
    
    c = 1/2 * (ur+ul)

    return - (1 - np.tanh((x - c*t)/(2*eps))**2)/(2*eps)    
    

xvec = np.linspace(0,1,1000)

param = {'eps': 1}


porder = 4
size = 30

mesh = mkmesh_1Dinterval(size, porder)
master = mkmaster1D(mesh, 4*porder)

uh_ini = initu_1D(mesh,exact_u)

qh_ini = initu_1D(mesh,exact_q)

uhath_ini = init_uhat1D(mesh,exact_u)

dt = 0.2

bc0 = exact_u(0)
bc1 = exact_u(1)

t = 1 

qhn1, uhn1, uhathn1 = BE_step(dt, t, exact_u, master, mesh, param, uh_ini, qh_ini, uhath_ini, 0, source = None)

uh_ini = initu_1D(mesh,exact_u)

uh_exact = exact_u(xvec,1.2)

fig1, ax1 = plt.subplots()

plot_1D(mesh,uh_ini,ax1, linestyle = 'dashed',label = 't=1, initial condtion')
plot_1D(mesh,uhn1,ax1, linestyle = 'dashdot', label = 't=1.2, BE')
ax1.plot(xvec,uh_exact,label='t=1.2, exact')
ax1.legend()

