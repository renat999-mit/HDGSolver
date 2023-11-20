import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from master import shape2d

__all__ = ['hdg_postprocess']



def hdg_postprocess(master, mesh, master1, mesh1, uh, qh):
    """
    hdg_postprocess postprocesses the hdg solution to obtain a better solution.
 
       master:       master structure of porder
       mesh:         mesh structure of porder
       master1:      master structure of porder+1
       mesh1:        mesh structure of porder+1
       uh:           approximate scalar variable
       qh:           approximate flux
       ustarh:       postprocessed scalar variable
    """

    qh = -qh
    
    ustarh = np.zeros((mesh1.dgnodes.shape[0], mesh1.t.shape[0]))
    
    nt = mesh.t.shape[0]
    
    shap1 = master1.shap[:,0,:]
    shapxi1 = master1.shap[:,1,:]
    shapet1 = master1.shap[:,2,:]
    shapxig1 = shapxi1 @ np.diag(master1.gwgh)
    shapetg1 = shapet1 @ np.diag(master1.gwgh)
    
    shape_0_1 = shape2d(master.porder, master.plocal, master1.gpts)
    
    for e in range(nt):
        
        dg1 = mesh1.dgnodes[:,:,e]
        
        xxi1 = dg1[:,0] @ shapxi1
        xet1 = dg1[:,0] @ shapet1
        yxi1 = dg1[:,1] @ shapxi1
        yet1 = dg1[:,1] @ shapet1
        jac1 = xxi1 * yet1 - xet1 * yxi1
        
        shapx1 =   shapxig1 @ np.diag(yet1) - shapetg1 @ np.diag(yxi1)
        shapy1 = - shapxig1 @ np.diag(xet1) + shapetg1 @ np.diag(xxi1)
        
        Mx = shapx1 @ np.diag(1/(master1.gwgh*jac1)) @ shapx1.T
        My = shapy1 @ np.diag(1/(master1.gwgh*jac1)) @ shapy1.T
        
        M = Mx + My
        
        uh1 = shape_0_1[:,0,:].T @ uh[:,e]
        qh1x = shape_0_1[:,0,:].T @ qh[:,0,e]
        qh1y = shape_0_1[:,0,:].T @ qh[:,1,e]
        
        rhsx = shapx1 @ qh1x
        rhsy = shapy1 @ qh1y
        
        rhs = rhsx + rhsy
        
        row_average = shap1 @ (master1.gwgh*jac1)
        average = (master1.gwgh*jac1) @ uh1
            
        M[-1,:] = row_average
        rhs[-1] = average
        
        ustarh[:,e] = np.linalg.solve(M,rhs)
        
    return ustarh

        
        
    