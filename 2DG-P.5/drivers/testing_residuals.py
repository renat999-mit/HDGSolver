#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:11:59 2023

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
from hdgker import hdg_solve, hdg_postprocess, DIRK_step, BE_step

porder = 6
ngrid  = 5

mesh = mkmesh_1Dinterval(ngrid, porder)
master = mkmaster1D(mesh, 2*porder)

print(mesh.t)
print(len(mesh.t))

eps = 1

param = {'eps': eps}

dbc = np.array([0.,0.])

def source(x):
    
    return np.pi**2*np.sin(np.pi*x) + np.pi*np.cos(np.pi*x)*np.sin(np.pi*x)

def exact(x):
    
    return np.sin(np.pi*x)

def exact_q(x):
    
    return np.pi*np.cos(np.pi*x)

uh0 = initu_1D(mesh,exact)

qh_init = initu_1D(mesh,exact_q)

uhath_init = init_uhat1D(mesh,exact)

fig, ax = plt.subplots()

plot_1D_uhat(mesh,uhath_init,ax)

plot_1D(mesh,uh0,ax)

def residuals(master, mesh, dbc, param, uh, qh, uhath, source = None, extra_src = None, alpha = None):
    
    npl = mesh.dgnodes.shape[0]
    nt  = mesh.t.shape[0]
    
    eps = param['eps']
    
    taud = len(mesh.t)*eps
    
    sh1d = np.squeeze(master.sh1d[:,0,:])
    shap1xi = np.squeeze(master.sh1d[:,1,:])
    
    H = np.zeros((nt+1,nt+1))
    r = np.zeros((nt+1,1))
    
    Jqq = np.zeros((npl,npl,nt))
    Jqu = np.zeros((npl,npl,nt))
    Jquhat = np.zeros((npl,2,nt))
    
    Juq = np.zeros((npl,npl,nt))
    Juu = np.zeros((npl,npl,nt))
    Juuhat = np.zeros((npl,2,nt))
    
    Juhatq = np.zeros((2,npl,nt))
    Juhatu = np.zeros((2,npl,nt))
    Juhatuhat = np.zeros((2,2,nt))
    
    r1 = np.zeros((npl,nt))
    r2 = np.zeros((npl,nt))
    
    for e in range(nt):
        
        # Define tau convection for the element based on local velocities
        tauc = 1.5*np.max(np.absolute(uh[:,e]))
        
        # Compute tau
        tau = taud + tauc
        
        dg = mesh.dgnodes[:,0,e]
        
        # Compute "Jacobian"
        xxi = shap1xi.T @ dg
        # dsdxi = np.sqrt(xxi**2)
        dsdxi = xxi
        
        # Get q and u at quadrature points
        qhg = sh1d.T @ qh[:,e]
        uhg = sh1d.T @ uh[:,e]
        
        # Define vector for "face" integrals
        uhath_int = np.zeros(npl)
        uhath_int[0] = -uhath[0,e] # here n = -1
        uhath_int[-1] = uhath[1,e] # here n = 1
        uhath_int = np.reshape(uhath_int,(npl,1))
        
        # Compute residuals
        
        qhg = np.reshape(qhg,(len(qhg),1))
        uhg = np.reshape(uhg,(len(uhg),1))
        
        r1[:,e] =   np.squeeze(sh1d @ np.diag(master.gw1d*dsdxi*1/eps) @ qhg + shap1xi @ np.diag(master.gw1d) @ uhg) \
                   - np.squeeze(uhath_int)  
                   
        # q equation
        Jqq[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * 1/eps) @ sh1d.T
    
        Jqu[:,:,e] = shap1xi @ np.diag(master.gw1d) @ sh1d.T
        
        Jquhat[0,0,e] = 1
        Jquhat[-1,-1,e] = -1
        
        r1_mat = Jqq[:,:,e] @ qh[:,e] + Jqu[:,:,e] @ uh[:,e] + Jquhat[:,:,e] @ uhath[:,e]
        
        print("r1_mat :", np.linalg.norm(r1_mat))
        
        uh_int = np.zeros(npl)
        uh_int[0] = -uh[0,e]
        uh_int[-1] = uh[-1,e]
        uh_int = np.reshape(uh_int,(npl,1))
        
        qh_int = np.zeros(npl)
        qh_int[0] = qh[0,e]
        qh_int[-1] = qh[-1,e]
        qh_int = np.reshape(qh_int,(npl,1))
        
        qhxg = np.squeeze(shap1xi.T @ qh[:,e])
        
        c_uhat = 0.5*uhath_int**2
        c_uhat[0] = -c_uhat[0]
        
        qh_int_aux = qh_int
        qh_int_aux[0] = - qh_int_aux[0]
        
        r2[:,e] = np.squeeze(sh1d @ np.diag(master.gw1d*dsdxi*alpha) @ uhg) - \
             np.squeeze(sh1d @ np.diag(master.gw1d) @ qhxg) - \
             np.squeeze(shap1xi @ np.diag(master.gw1d) @ (0.5*uhg**2)) + \
             np.squeeze(tau*uh_int) + \
             np.squeeze(c_uhat) - \
             np.squeeze(tau*uhath_int)
             
        # u equation
        
        Juq[:,:,e] = -sh1d @ np.diag(master.gw1d) @ shap1xi.T
        
        
        Juu[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * alpha) @ sh1d.T
        
        Juu[:,:,e] -= shap1xi @ np.diag(master.gw1d * uhg) @ sh1d.T
        Juu[0,0,e] -= tau
        Juu[-1,-1,e] += tau
        
        Juuhat[0,0,e] = -uhath[0,e] + tau
        Juuhat[-1,-1,e] = uhath[1,e] - tau
        
        if source is not None:
            
            sg = sh1d.T @ source(dg)
        
            r2[:,e] += np.squeeze(- sh1d @ np.diag(master.gw1d * dsdxi) @ sg)
            
        if extra_src is not None:
            
            extra_srcg = sh1d.T @ extra_src[:,e]
            
            r2[:,e] += np.squeeze(- sh1d @ np.diag(master.gw1d * dsdxi) @ extra_srcg)
            
        c_int = np.zeros(npl)
        c_int[0] = 0.5*uhath[0,e]**2
        c_int[-1] = 0.5*uhath[-1,e]**2
        c_int = np.reshape(c_int,(npl,1))
        
        # Local residual for uhath
        
        q_vec = np.array([-qh[0,e],qh[-1,e]])
        
        u_vec = np.array([-tau*uh[0,e],tau*uh[-1,e]])
        
        c_vec = np.array([-0.5*uhath[0,e]**2,0.5*uhath[1,e]**2])
        
        uhat_vec = np.array([-tau*uhath[0,e],tau*uhath[1,e]])
        
        r3 = q_vec - u_vec - c_vec + uhat_vec
        
        Juhatq[0,0,e] = -1
        Juhatq[-1,-1,e] = 1
        
        Juhatu[0,0,e] = tau
        Juhatu[-1,-1,e] = -tau
        
        Juhatuhat[0,0,e] = -tau + (+uhath[0,e])
        Juhatuhat[-1,-1,e] = tau + (-uhath[1,e])
        
        
        print(f"Element {e}: ")
        print(f"rq = ", np.linalg.norm(r1[:,e]))
        print(f"ru = ", np.linalg.norm(r2[:,e]))
        # print(f"ruhat = ", np.linalg.norm(r3))
        
        
        # r3 = np.squeeze(np.array([qh_int[0],qh_int[-1]]) - \
        #      tau*np.array([-uh[0,e],uh[-1,e]]) - \
        #      np.array([c_int[0],c_int[-1]]) + \
        #      tau*np.array([-uhath[0,e],uhath[1,e]]))
            
        # Compute matrices
            
        # # q equation
        # Jqq[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * 1/eps) @ sh1d.T
    
        # Jqu[:,:,e] = sh1d @ np.diag(master.gw1d) @ shap1xi.T
        
        # Jquhat[0,0,e] = -1
        # Jquhat[-1,-1,e] = 1
        
        # u equation
        
        # Juq[:,:,e] = -sh1d @ np.diag(master.gw1d) @ shap1xi.T
        # # Juq[0,0,e] -= -1
        # # Juq[-1,-1,e] -= 1
        
        # Juu[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * alpha) @ sh1d.T
        
        # Juu[:,:,e] -= sh1d @ np.diag(master.gw1d* (uhg)) @ shap1xi.T
        # Juu[0,0,e] -= tau
        # Juu[-1,-1,e] += tau
        
        # Juuhat[0,0,e] = uhath[0,e] + tau
        # Juuhat[-1,-1,e] = uhath[1,e] - tau
        
        # uhat equation
        
        # Juhatq[0,0,e] = 1
        # Juhatq[-1,-1,e] = 1
        
        # Juhatu[0,0,e] = -tau
        # Juhatu[-1,-1,e] = tau
        
        # Juhatuhat[0,0,e] = -tau + (-uhath[0,e])
        # Juhatuhat[-1,-1,e] = - tau + (uhath[1,e])
        
        # Compute elemental matrix H_k and vector r_k
        
        aux_mat = np.block([[Jqq[:,:,e],Jqu[:,:,e]],[Juq[:,:,e],Juu[:,:,e]]])
        aux_rhs1 = np.vstack([Jquhat[:,:,e],Juuhat[:,:,e]])
        aux_rhs2 = np.vstack([-r1[:,e],-r2[:,e]])
        aux_rhs2 = np.reshape(aux_rhs2,(2*npl,1))
        
        H_k = Juhatuhat[:,:,e] - np.hstack([Juhatq[:,:,e],Juhatu[:,:,e]]) @ np.linalg.solve(aux_mat,aux_rhs1)
        r_k = -r3 - np.hstack([Juhatq[:,:,e],Juhatu[:,:,e]]) @ np.linalg.solve(aux_mat,np.squeeze(aux_rhs2))
        r_k = np.reshape(r_k,(2,1))
        
        # Assemble into global matrix H and vector r
        
        H[e:e+2,e:e+2] += H_k
        r[e:e+2] = r[e:e+2] + r_k
        
    # Apply Dirichlet BC
    
    H[0,:] = np.zeros(nt+1)
    H[0,0] = 1.
    r[0] = 0
    
    H[-1,:] = np.zeros(nt+1)
    H[-1,-1] = 1.
    r[-1] = 0
    
    print(r)
    
    print("Global r_uhat: ", np.linalg.norm(r))
        
        
        
            
        
        
residuals(master, mesh, dbc, param, uh0, qh_init, uhath_init, source, None,0)

def func(x):
    
    return x**2

def integral(mesh,master, func):
    
    nt = len(mesh.t)
    
    sh1d = np.squeeze(master.sh1d[:,0,:])
    shap1xi = np.squeeze(master.sh1d[:,1,:])
    
    result = 0
    for e in range(nt):
    
        dg = mesh.dgnodes[:,0,e]
        
        xxi = shap1xi.T @ dg
        dsdxi = np.sqrt(xxi**2)
        
        xg = sh1d.T @ dg
        
        funcg = func(xg)
        
        result += funcg @ (master.gw1d*dsdxi)
    
    return result

x_2 = integral(mesh,master,func) - 1/3