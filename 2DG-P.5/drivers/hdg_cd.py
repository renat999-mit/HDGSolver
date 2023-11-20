# Driver for Contaminant Dispersion Problem

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt

from mesh import *
from util import *
from master import *
from hdgker import hdg_solve, hdg_postprocess

def diff_square_convergence(size, porder, taud, save = False, name = None):
    
    error_ustarh = np.zeros((len(size),len(porder)))
    error_uh = np.zeros((len(size),len(porder)))
    error_qh = np.zeros((len(size),len(porder)))
    
    kappa = 1.0
    c = [0.0, 0.0]

    param = {'kappa': kappa, 'c': c}
    
    exactu = lambda p: np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1])
    exactqx = lambda p: -np.cos(np.pi*p[:,0])*np.sin(np.pi*p[:,1])*np.pi
    exactqy = lambda p: -np.sin(np.pi*p[:,0])*np.cos(np.pi*p[:,1])*np.pi
    
    source = lambda p: 2*(np.pi**2)*np.sin(np.pi*p[:,0])*np.sin(np.pi*p[:,1]) # f = 2*pi^2*sin(pi*x)*sin(pi*y)
    dbc    = lambda p: np.zeros((p.shape[0],1))

    for p in range(len(porder)):
        
        print(f'p = {porder[p]}')
        
        for i in range(len(size)):
            
            print(f'size = {size[i]}')
            
            wig = 0.0       # amopunt of mesh distortion
        
            mesh = mkmesh_square(size[i], size[i], porder[p])
            mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
            master = mkmaster(mesh, 2*porder[p])
        
            tauds = [taud[i],taud[i]] # different value of tau_d on the boundary and interior faces
            
            # HDG Solution
            uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, tauds)
        
            # HDG postprocessing
            mesh1   = mkmesh_square(size[i], size[i], porder[p]+1)
            mesh1   = mkmesh_distort(mesh1, wig)
            master1 = mkmaster(mesh1, 2*(porder[p]+1))
            ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)
        
            # calculate errors
            error_ustarh[i,p] = np.sqrt(l2_error(mesh1, ustarh, exactu))
            error_uh[i,p] = np.sqrt(l2_error(mesh, uh, exactu))
            
            error_qhx2 = l2_error(mesh, qh[:,0,:], exactqx)
            error_qhy2 = l2_error(mesh, qh[:,1,:], exactqy)
            
            error_qh[i,p] = np.sqrt(error_qhx2 + error_qhy2)
            
    size = 1/size
    
    def p_line(h,err):
        
        fit = np.polyfit(np.log(h),np.log(err),1)
        
        return np.exp(fit[1])*h**fit[0], fit[0]
    
        
    fig1, ax1 = plt.subplots()
    
    ax1.set_title(r'$u^*_h$')
    for p in range(len(porder)):
        
        p_, conv_order = p_line(size,error_ustarh[:,p])
        
        ax1.loglog(size,error_ustarh[:,p],'o', label=f'p={porder[p]}, order = {conv_order:.2f}')
        ax1.loglog(size,p_,ls= 'dashed', color=ax1.lines[-1].get_color())
        
    ax1.set_xlabel("log h")
    ax1.set_ylabel(r"log $||e||_{2}$")
    ax1.legend()
    plt.show()
    
    fig2, ax2 = plt.subplots()
    
    ax2.set_title(r'$u_h$')
    for p in range(len(porder)):
        
        p_, conv_order = p_line(size,error_uh[:,p])
        
        ax2.loglog(size,error_uh[:,p],'o', label=f'p={porder[p]}, order = {conv_order:.2f}')
        ax2.loglog(size,p_,ls= 'dashed', color=ax2.lines[-1].get_color())
        
    ax2.set_xlabel("log h")
    ax2.set_ylabel(r"log $||e||_{2}$")
    ax2.legend()
    plt.show()
    
    fig3, ax3 = plt.subplots()
    
    ax3.set_title(r'$q_h$')
    for p in range(len(porder)):
        
        p_, conv_order = p_line(size,error_qh[:,p])
        
        ax3.loglog(size,error_qh[:,p],'o', label=f'p={porder[p]}, order = {conv_order:.2f}')
        ax3.loglog(size,p_,ls= 'dashed', color=ax3.lines[-1].get_color())
        
    ax3.set_xlabel("log h")
    ax3.set_ylabel(r"log $||e||_{2}$")
    ax3.legend()
    plt.show()
    
    
    if save:
        
        fig1.savefig(f"{name[0]}.pdf")
        fig2.savefig(f"{name[1]}.pdf")
        fig3.savefig(f"{name[2]}.pdf")

if __name__ == "__main__":
    
    porder = 2
    ngrid  = 2
    wig = 0.0       # amopunt of mesh distortion

    mesh = mkmesh_square(ngrid, ngrid, porder)
    mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)

    kappa = 2.0
    c = [100.0, 50.0]

    param = {'kappa': kappa, 'c': c}
    source = lambda p: 10.0*np.ones((p.shape[0],1))
    dbc    = lambda p: np.zeros((p.shape[0],1))
    
    tau_val = [kappa,kappa]

    # HDG Solution
    uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, tau_val)

    # HDG postprocessing
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))
    ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)

    fig, axs = plt.subplots(2)
    
    pause = lambda : input('(press enter to continue)')
    plt.ion()
    scaplot_raw(axs[0], mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
    scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    
    porder = np.array([1,2,3,4])
    size = np.array([4,8,16])
    
    h = 1/size
    
    taud = 1/h
    
    names = ['ustarh_1_h','uh_1_h','qh_1_h']
    
    diff_square_convergence(size, porder, taud, save = True, name = names)
    

    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot(mesh, uh, show_mesh=False, pplot=porder+2, interactive=True, title='HDG Solution')
    # pause()
    # scaplot(mesh1, ustarh, show_mesh=False, pplot=porder+3, interactive=True, title='Postprocessed Solution')
    # pause()  

        


  