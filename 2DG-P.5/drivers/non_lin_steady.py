#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:55:59 2023

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


size = np.array([7,14,28])
porder = np.array([2,3,4,5])

error_ustarh = np.zeros((len(size),len(porder)))
error_uh = np.zeros((len(size),len(porder)))
error_qh = np.zeros((len(size),len(porder)))

eps = 1

param = {'eps': eps}

dbc = np.array([0.,0.])

alpha = 0

extra_src = None

def source(x):
    
    return np.pi**2*np.sin(np.pi*x) + np.pi*np.cos(np.pi*x)*np.sin(np.pi*x)

def exact(x):
    
    return np.sin(np.pi*x)

def exact_q(x):
    
    return np.pi*np.cos(np.pi*x)

def initial_guess(x):
    
    return np.zeros(len(x))


for p in range(len(porder)):
    
    print(f'p = {porder[p]}')
    
    for i in range(len(size)):
        
        print(f'size = {size[i]}')
        
        mesh = mkmesh_1Dinterval(size[i], porder[p])
        master = mkmaster1D(mesh, 4*porder[p])
        
        uh_ini = initu_1D(mesh,initial_guess)

        qh_ini = initu_1D(mesh,initial_guess)

        uhath_ini = init_uhat1D(mesh,initial_guess)
    
        # HDG Solution
        qh, uh, uhath = newton_method(master, mesh, dbc, param, uh_ini, qh_ini, uhath_ini, source, extra_src, alpha)
    
        error_uh_ = np.sqrt(l2_norm_1D(mesh,uh,exact))
        print(error_uh_)
    
        # calculate errors
        error_uh[i,p] = np.sqrt(l2_norm_1D(mesh,uh,exact))
        
        error_qh[i,p] = np.sqrt(l2_norm_1D(mesh,qh,exact_q))
 
#%%        
size = 1/size

def p_line(h,err):
    
    fit = np.polyfit(np.log(h),np.log(err),1)
    
    return np.exp(fit[1])*h**fit[0], fit[0]

fig2, ax2 = plt.subplots()

ax2.set_title(r'$u_h$')
for p in range(len(porder)):
    
    p_, conv_order = p_line(size,error_uh[:,p])
    
    ax2.loglog(size,error_uh[:,p],'o', label=f'$p=${porder[p]}, order = {conv_order:.2f}')
    ax2.loglog(size,p_,ls= 'dashed', color=ax2.lines[-1].get_color())
    
ax2.set_xlabel("$\log h$")
ax2.set_ylabel(r"$\log ||e||_{2}$")
ax2.legend()
fig2.savefig("non_lin_convergence_u.pdf")
plt.show()

fig3, ax3 = plt.subplots()

ax3.set_title(r'$q_h$')
for p in range(len(porder)):
    
    p_, conv_order = p_line(size,error_qh[:,p])
    
    ax3.loglog(size,error_qh[:,p],'o', label=f'$p=${porder[p]}, order = {conv_order:.2f}')
    ax3.loglog(size,p_,ls= 'dashed', color=ax3.lines[-1].get_color())
    
ax3.set_xlabel(r"$\log h$")
ax3.set_ylabel(r"$\log ||e||_{2}$")
ax3.legend()
fig3.savefig("non_lin_convergence_q.pdf")
plt.show()
        

    
    

