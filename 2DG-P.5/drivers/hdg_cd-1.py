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

if __name__ == "__main__":
    porder = 3
    ngrid  = 4
    wig = 0.1      # amount of mesh distortion

    mesh = mkmesh_square(ngrid, ngrid, porder)
    mesh = mkmesh_distort(mesh, wig)            # Mesh distortion
    master = mkmaster(mesh, 2*porder)

    kappa = 1.0
    c = [100, 100]

    param = {'kappa': kappa, 'c': c}
    source = lambda p: 10.0*np.ones((p.shape[0],1))
    dbc    = lambda p: np.zeros((p.shape[0],1))

    # HDG Solution
    uh, qh, uhath = hdg_solve(master, mesh, source, dbc, param, [1,1])

    # HDG postprocessing
    mesh1   = mkmesh_square(ngrid, ngrid, porder+1)
    mesh1   = mkmesh_distort(mesh1, wig)
    master1 = mkmaster(mesh1, 2*(porder+1))
    
    
    ustarh  = hdg_postprocess(master, mesh, master1, mesh1, uh, qh/kappa)

    fig, axs = plt.subplots()
    
    pause = lambda : input('(press enter to continue)')
    plt.ion()
    scaplot_raw(axs, mesh, uh, show_mesh=True, pplot=porder+2, title='HDG Solution')
    
    fig.savefig(f'uh_c_{c[0]}_{ngrid}.pdf')
    # scaplot_raw(axs[1], mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    
    fig1, ax1 = plt.subplots()

    scaplot_raw(ax1, mesh1, ustarh, show_mesh=True, pplot=porder+3, title='Postprocessed Solution')
    
    fig1.savefig(f'ustarh_c_{c[0]}_{ngrid}.pdf')
    # pause = lambda : input('(press enter to continue)')
    # plt.ion()
    # scaplot(mesh, uh, show_mesh=False, pplot=porder+2, interactive=True, title='HDG Solution')
    # pause()
    # scaplot(mesh1, ustarh, show_mesh=False, pplot=porder+3, interactive=True, title='Postprocessed Solution')
    # pause()  

    print(len(master.gw1d))
        


  