#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:30:37 2023

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
from hdgker import hdg_solve, hdg_postprocess, DIRK_step, BE_step, newton_method, BDF2_step, BDF3_step, BDF4_step

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

size = np.array([10,20,40])
dt_ = np.array(([0.4,0.2,0.1])) 
porder = np.array([1,2,3,4,5,6])
t_fin_ = np.array([7.2,7.2,7.2,7.2,7.2,7.2])

error_ustarh = np.zeros((len(size),len(porder)))
error_uh = np.zeros((len(size),len(porder)))
error_qh = np.zeros((len(size),len(porder)))

param = {'eps': 1}

for p in range(len(porder)):
    
    print(f'p = {porder[p]}')
    
    for i in range(len(size)):
        
        print(f'size = {size[i]}')
        
        mesh = mkmesh_1Dinterval(size[i], porder[p])
        master = mkmaster1D(mesh, 4*porder[p])
        
        uh_ini = initu_1D(mesh,exact_u)
        
        qh_ini = initu_1D(mesh,exact_q)
        
        uhath_ini = init_uhat1D(mesh,exact_u)
        
        t = 1
        
        t_fin = t_fin_[p]
        
        dt = dt_[i]
        
        uhist = np.zeros((len(mesh.plocal),len(mesh.t),4))
        
        step_counter = 0
        
        qh, uh, uhath = qh_ini, uh_ini, uhath_ini
        
        while (t_fin - t) > 1e-10:
            
            if step_counter == 0:
                
                qhn1, uhn1, uhathn1 = BE_step(dt, t, exact_u, master, mesh, param, uh, qh, uhath, uhist, source = None)
                step_counter += 1
                
            elif step_counter == 1:
                
                qhn1, uhn1, uhathn1 = BE_step(dt, t, exact_u, master, mesh, param, uh, qh, uhath, uhist, source = None)
                step_counter += 1
                
            elif step_counter == 2:
                
                qhn1, uhn1, uhathn1 = BDF2_step(dt, t, exact_u, master, mesh, param, uh, qh, uhath, uhist, source = None)
                step_counter += 1
                
            elif step_counter == 3:
                
                qhn1, uhn1, uhathn1 = BDF3_step(dt, t, exact_u, master, mesh, param, uh, qh, uhath, uhist, source = None)
                step_counter += 1
                
            elif step_counter >= 4:
                
                qhn1, uhn1, uhathn1 = BDF4_step(dt, t, exact_u, master, mesh, param, uh, qh, uhath, uhist, source = None)
                
            uhist[:,:,1:4] = uhist[:,:,0:3]
            
            uhist[:,:,0] = uhn1
            
            t += dt
            
            print(t)
            
            qh, uh, uhath = qhn1, uhn1, uhathn1
            
        def exact_u_fin(x, eps = 1):
            
            ul = 2
            ur = 0
            
            c = 1/2 * (ur+ul)
            
            return 1 - np.tanh((x - c*t)/(2*eps))
        
        def exact_q_fin(x,eps = 1):
            
            ul = 2
            ur = 0
            
            c = 1/2 * (ur+ul)

            return - (1 - np.tanh((x - c*t)/(2*eps))**2)/(2*eps) 
            
        error_uh_ = np.sqrt(l2_norm_1D(mesh,uh,exact_u_fin))
        print(error_uh_)
    
        # calculate errors
        error_uh[i,p] = np.sqrt(l2_norm_1D(mesh,uh,exact_u_fin))
        
        error_qh[i,p] = np.sqrt(l2_norm_1D(mesh,qh,exact_q_fin))
    

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
    
ax2.set_xlabel(r"$\log h$")
ax2.set_ylabel(r"$\log ||e||_{2}$")
ax2.legend()
fig2.savefig("burgers_BDF4_u.pdf")
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
fig3.savefig('burgers_BDF4_q.pdf')
plt.show()
        
#%%
uh_ini = initu_1D(mesh,exact_u)

uh_exact = exact_u(xvec,t)

fig1, ax1 = plt.subplots()

plot_1D(mesh,uh_ini,ax1, linestyle = 'dashed',label = 't=1, initial condtion')
plot_1D(mesh,uhn1,ax1, linestyle = 'dashdot', label = f't={t:.2f}, Numerical')
ax1.plot(xvec,uh_exact,label=f't={t:.2f}, exact')
ax1.legend()    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        