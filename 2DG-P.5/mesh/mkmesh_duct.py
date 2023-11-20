import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import distmesh as dm
from mesh import Mesh, mkt2f, setbndnbrs, createnodes, squaremesh, cgmesh
from util import *
from master import *

__all__ = ['mkmesh_duct']
    
def mkmesh_duct(mesh, db, dt, H):
    """
    tfi_duct map a unit square mesh to a cos^2 duct
    mesh = mkmesh_duct(mesh,db,dt,h)

       mesh:     mesh structure generated by mkamesh_square
                  flag = 0 (diagonals sw - ne) (default)
                  flag = 1 (diagonals nw - se)
       db:       height of bottom bump
       dt:       height of top bump
       h:        height of channel
    """

    p = mesh.p
    mesh.fcurved[:] = True
    mesh.tcurved[:] = True
    
    pnew = np.zeros_like(p)
    pnew[:,0] = 3.0 * p[:,0]
    ii = np.where((pnew[:,0]<=1.0) | (pnew[:,0]>=2.0))
    pnew[ii,1] = H * p[ii,1]
    ii = np.where((pnew[:,0]>1.0) & (pnew[:,0]<2.0))
    pnew[ii,1] = p[ii,1]*(H-dt*np.sin(np.pi*(pnew[ii,0]-1.0))**2) + \
                (1.0-p[ii,1])*(db*np.sin(np.pi*(pnew[ii,0]-1.0))**2)
    mesh.p = pnew

    p = np.reshape(np.moveaxis(mesh.dgnodes, 2, 1), (-1,2))
    pnew = np.zeros(p.shape)

    pnew[:,0] = 3.0 * p[:,0]
    ii = np.where((pnew[:,0]<=1.0) | (pnew[:,0]>=2.0))
    pnew[ii,1] = H * p[ii,1]
    ii = np.where((pnew[:,0]>1.0) & (pnew[:,0]<2.0))
    pnew[ii,1] = p[ii,1]*(H-dt*np.sin(np.pi*(pnew[ii,0]-1.0))**2) + \
                (1.0-p[ii,1])*(db*np.sin(np.pi*(pnew[ii,0]-1.0))**2)

    mesh.dgnodes = np.moveaxis(np.reshape(pnew, (-1, mesh.t.shape[0], 2)), 1,2)

    mesh = cgmesh(mesh)

    return mesh

if __name__ == "__main__":
    mesh = mkmesh_duct( 12, 4, 3, 0, 0.2, 0.1, 1.0)
    print("p", mesh.p)
    print("t", mesh.t)
    print("t2f", mesh.t2f)
    print("f", mesh.f)
    meshplot_curved(mesh, nodes=True, pplot=5)