import numpy as np

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from mesh import Mesh1D, mkt2f_1D, setbndnbrs, createnodes_1D, interval_1D, cgmesh
from master import *
from master.master import localpnts1d, tlocal1d, Master1D, mkmaster1D
from util import *


__all__ = ['mkmesh_1Dinterval']

def mkmesh_1Dinterval(m=2, porder=1):
    """ 
    mkmesh_1Dinterval creates 1d mesh data structure for unit interval.
    mesh=mkmesh_square(m,porder)
 
       mesh:      mesh structure
       m:         number of points 
       porder:    polynomial order of approximation (default=1)
 
    see also: squaremesh, mkt2f, setbndnbrs, uniformlocalpnts, createnodes
    """
    assert m >= 2
    
    p, t = interval_1D(m)
    f = mkt2f_1D(t)

    plocal = localpnts1d(porder)
    tlocal = tlocal1d(porder)

    mesh = Mesh1D(p, t, f, porder, plocal, tlocal)
    
    mesh = createnodes_1D(mesh)
    # mesh = cgmesh(mesh)

    return mesh

if __name__ == "__main__":
    mesh = mkmesh_1Dinterval(6,4)
    # meshplot(mesh, True, 'pt')
    print("p", mesh.p)
    # print("t", mesh.t)
    # print("f", mesh.f)
    # print("first element nodes: ", mesh.dgnodes[:,0,0])
    # print("second element nodes: ", mesh.dgnodes[:,0,1])
    # print("last element nodes: ", mesh.dgnodes[:,0,-1])
    
    master = mkmaster1D(mesh,2*4)
    
    # print("pcg", mesh.pcg)
    # print("tcg", mesh.tcg)

    # dgnodes = np.empty(mesh.dgnodes.shape, dtype=float)
    # for i in range(mesh.t.shape[0]):
    #     dgnodes[:,:,i] = mesh.pcg[mesh.tcg[i,:],:]

    # mesh.dgnodes = dgnodes
    # meshplot(mesh, True, annotate='pt')