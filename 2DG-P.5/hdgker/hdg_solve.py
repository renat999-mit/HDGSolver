import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from util import *
import matplotlib.pyplot as plt

from master import shape2d

__all__ = ['hdg_solve']

def localprob(dg, master, m, source, param, bf, tau_val):
    """
    localprob solves the local convection-diffusion problems for the hdg method
       dg:              dg nodes
       master:          master element structure
       m[3*nps, ncol]:  values of uhat at the element edges (ncol is the
                        number of right hand sides. i.e a different local 
                        problem is solved for each right hand side)
       source:          source term
       param:           param['kappa']= diffusivity coefficient
                        param['c'] = convective velocity
       umf[npl, ncol]:  uh local solution
       qmf[npl,2,ncol]: qh local solution
    """
    porder = master.porder

    kappa = param['kappa']
    c     = param['c']
    # taud  = kappa
    
    taud_interior = tau_val[0]
    taud_boundary = tau_val[1]

    nps   = porder+1
    ncol  = m.shape[1]
    npl   = dg.shape[0]

    perm = master.perm[:,:,0]

    qmf = np.zeros((npl, 2, ncol))
    
    Fx = np.zeros((npl, ncol))
    Fy = np.zeros((npl, ncol))
    Fu = np.zeros((npl, ncol))
    
    # Volume integral
    shap = master.shap[:,0,:]
    shapxi = master.shap[:,1,:]
    shapet = master.shap[:,2,:]
    shapxig = shapxi @ np.diag(master.gwgh)
    shapetg = shapet @ np.diag(master.gwgh)

    xxi = dg[:,0] @ shapxi
    xet = dg[:,0] @ shapet
    yxi = dg[:,1] @ shapxi
    yet = dg[:,1] @ shapet
    jac = xxi * yet - xet * yxi
    shapx =   shapxig @ np.diag(yet) - shapetg @ np.diag(yxi)
    shapy = - shapxig @ np.diag(xet) + shapetg @ np.diag(xxi)
    M  = (shap @ np.diag(master.gwgh * jac) @ shap.T)/kappa
    Cx = shap @ shapx.T
    Cy = shap @ shapy.T

    D = -c[0]*Cx.T - c[1]*Cy.T

    if source:
        pg = shap.T @ dg
        src = source( pg)
        Fu = shap @ np.diag(master.gwgh*jac) @ src
    
    Fu = np.reshape(Fu,(npl,ncol))
    
    sh1d = np.squeeze(master.sh1d[:,0,:])
    for s in range(3):
        
        if s in bf:
            
            taud = taud_boundary
            
        else:
            
            taud = taud_interior
            
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2)
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
        
        cnl = c[0]*nl[:,0] + c[1]*nl[:,1]
    
        tauc = np.abs(cnl)
        tau  = taud + tauc

        D[np.ix_(perm[:,s], perm[:,s])] = D[np.ix_(perm[:,s], perm[:,s])] + sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T
    
    for s in range(3):
        
        if s in bf:
            
            taud = taud_boundary
            
        else:
            
            taud = taud_interior
        
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2)
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
       
        for icol in range(ncol):     # Loop over all the right-hand-sides
            ml = m[s*nps:(s+1)*nps, icol]
       
            cnl = c[0]*nl[:,0] + c[1]*nl[:,1]
    
            tauc = np.abs(cnl)
            tau  = taud + tauc
    
            Fx[perm[:,s], icol] = Fx[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T @ ml
            Fy[perm[:,s], icol] = Fy[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T @ ml

            Fu[perm[:,s], icol] = Fu[perm[:,s], icol] - sh1d @ np.diag(master.gw1d*dsdxi*(cnl-tau)) @ sh1d.T @ ml

    M1Fx = np.linalg.solve(M, Fx)
    M1Fy = np.linalg.solve(M, Fy)

    M1   = np.linalg.inv(M)
    
    umf = np.linalg.solve(D + Cx @ M1 @ Cx.T + Cy @ M1 @ Cy.T, Fu - Cx @ M1Fx - Cy @ M1Fy)
    qmf[:,0,:] = M1Fx + np.linalg.solve(M, Cx.T @ umf)
    qmf[:,1,:] = M1Fy + np.linalg.solve(M, Cy.T @ umf)

    return umf, qmf


def elemmat_hdg(dg, master, source, param, bf, tau_val):
    """
    elemmat_hdg calcualtes the element and force vectors for the hdg method
 
       dg:              dg nodes
       master:          master element structure
       source:          source term
       param:           param['kappa']   = diffusivity coefficient
                        param['c'] = convective velocity
       ae[3*nps,3*nps]: element matrix (nps is nimber of points per edge)
       fe[3*nps,1]:     element forcer vector
    """
    nps   = master.porder + 1
    npl   = dg.shape[0]
    
    perm = master.perm[:,:,0]
    
    kappa = param['kappa']
    c     = param['c']
    # taud  = kappa
    taud_interior = tau_val[0]
    taud_boundary = tau_val[1]

    mu = np.identity(3*nps)
    um0, qm0 = localprob(dg, master, mu, None, param, bf, tau_val)
    
      
    m = np.zeros((3*nps,1))
    u0f, q0f = localprob(dg, master, m, source, param, bf, tau_val)
    
    M = np.zeros((3*nps,3*nps))
    
    Cx = np.zeros((3*nps,npl))
    Cy = np.zeros((3*nps,npl))
    
    E = np.zeros((3*nps,npl))
    
    sh1d = np.squeeze(master.sh1d[:,0,:])
    
    for s in range(3):
        
        if s in bf:
            
            taud = taud_boundary
            
        else:
            
            taud = taud_interior
        
        xxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 0]
        yxi = np.squeeze(master.sh1d[:,1,:]).T @ dg[perm[:,s], 1]
        dsdxi = np.sqrt(xxi**2 + yxi**2)
        nl = np.column_stack((yxi/dsdxi, -xxi/dsdxi))
        
        cnl = c[0]*nl[:,0] + c[1]*nl[:,1]
    
        tauc = np.abs(cnl)
        tau  = taud + tauc
        
        idx1dSav = np.arange(nps*s,nps*s+nps)
        
        M[np.ix_(idx1dSav,idx1dSav)] = sh1d @ np.diag(master.gw1d*dsdxi*(tau-cnl)) @ sh1d.T
        
        Cx[np.ix_(idx1dSav,perm[:,s])] = sh1d @ np.diag(master.gw1d*dsdxi*nl[:,0]) @ sh1d.T
        Cy[np.ix_(idx1dSav,perm[:,s])] = sh1d @ np.diag(master.gw1d*dsdxi*nl[:,1]) @ sh1d.T
        
        E[np.ix_(idx1dSav,perm[:,s])] = sh1d @ np.diag(master.gw1d*dsdxi*tau) @ sh1d.T
    
    ae = M - np.hstack([Cx,Cy,E]) @ np.vstack([qm0[:,0,:],qm0[:,1,:],um0])
    
    fe = np.squeeze(np.hstack([Cx,Cy,E]) @ np.vstack([q0f[:,0,:],q0f[:,1,:],u0f]))
    
    return ae, fe


def hdg_solve(master, mesh, source, dbc, param, tau_val):
    """
    hdg_solve solves the convection-diffusion equation using the hdg method.
    [uh,qh,uhath]=hdg_solve(mesh,master,source,dbc,param)
 
       master:       master structure
       mesh:         mesh structure
       source:       source term
       dbc:          dirichlet data 
       param:        param['kappa']   = diffusivity coefficient
                     param['c'] = convective velocity
       uh:           approximate scalar variable
       qh:           approximate flux
       uhath:        approximate trace                              
    """

    nps = mesh.porder + 1
    npl = mesh.dgnodes.shape[0]
    nt  = mesh.t.shape[0]
    nf  = mesh.f.shape[0]

    ae  = np.zeros((3*nps, 3*nps, nt))
    fe  = np.zeros((3*nps, nt))
    
    boundary_faces = np.squeeze(np.where(mesh.f[:,3] < 0))

    for i in range(nt):
        
        faces = np.absolute(mesh.t2f[i]) - 1
        
        bf = []
        for j in range(len(faces)):
            
            if faces[j] in boundary_faces:
                
                bf.append(j)
                
        ae[:,:,i], fe[:,i] = elemmat_hdg( mesh.dgnodes[:,:,i], master, source, param, bf, tau_val)
        
    
    elcon = np.zeros((3*nps,nt),dtype=int)
    
    for e in range(nt):
        
        for s in range(3):
            
            reverse = False
            
            f = mesh.t2f[e,s]
            
            if f < 0:
                
                reverse = True
                
            f = abs(f) - 1
            
            global_nodes = np.arange(f*nps,(f+1)*nps,dtype=int)
            
            if reverse:
                
                global_nodes = global_nodes[::-1]
                
            elcon[s*nps:(s+1)*nps,e] = global_nodes
    
    H = np.zeros((nps*mesh.f.shape[0],nps*mesh.f.shape[0]))
    R = np.zeros(nps*mesh.f.shape[0])
    
    for e in range(nt):
        
        H[np.ix_(elcon[:,e], elcon[:,e])] += ae[:,:,e]
        
        R[elcon[:,e]] += fe[:,e]
    
    #Apply Dirichlet BC
    
    for f in boundary_faces:
        
        e, local_f = np.where(np.absolute(mesh.t2f)-1 == f)
        
        global_nodes = np.arange(f*nps,(f+1)*nps,dtype=int)
        
        if mesh.t2f[e,local_f] < 0:
            
            local_nodes = master.perm[:,local_f,1]
            
            global_nodes = global_nodes[::-1]
            
            
        else:
            
            local_nodes = master.perm[:,local_f,0]
            
        dirichlet_data = dbc(mesh.dgnodes[local_nodes,:,e])
        
        H[global_nodes,:] = np.zeros((nps,H.shape[1]))
        
        for node in global_nodes:
            
            H[node,node] = 1

        R[global_nodes] = np.squeeze(dirichlet_data)
    
    uhath = spsolve(H,R)
    uhath = np.reshape(uhath, (len(uhath),1))
    
    uh = np.zeros((npl,nt))
    qh = np.zeros((npl,2,nt))
    
    for e in range(nt):
        
        faces = np.absolute(mesh.t2f[e]) - 1
        
        bf = []
        for j in range(len(faces)):
            
            if faces[j] in boundary_faces:
                
                bf.append(j)
                
        result_u, result_q = localprob(mesh.dgnodes[:,:,e], master, uhath[elcon[:,e],:], source, param, bf, tau_val)  
        uh[:,e] = np.squeeze(result_u)
        qh[:,0,e] = np.squeeze(result_q[:,0])
        qh[:,1,e] = np.squeeze(result_q[:,1])
        
    return uh, qh, uhath

def hdg_solve1D(master, mesh, dbc, param, uh, qh, uhath, source = None, extra_src = None, alpha = None):
    
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
        
        
        # r3 = np.squeeze(np.array([qh_int[0],qh_int[-1]]) - \
        #      tau*np.array([-uh[0,e],uh[-1,e]]) - \
        #      np.array([c_int[0],c_int[-1]]) + \
        #      tau*np.array([-uhath[0,e],uhath[1,e]]))
            
        # Compute matrices
            
        # q equation
        
        Jqq[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * 1/eps) @ sh1d.T
    
        Jqu[:,:,e] = shap1xi @ np.diag(master.gw1d) @ sh1d.T
        
        Jquhat[0,0,e] = 1
        Jquhat[-1,-1,e] = -1
        
        # u equation
        
        Juq[:,:,e] = -sh1d @ np.diag(master.gw1d) @ shap1xi.T
        
        
        Juu[:,:,e] = sh1d @ np.diag(master.gw1d * dsdxi * alpha) @ sh1d.T
        
        Juu[:,:,e] -= shap1xi @ np.diag(master.gw1d * np.squeeze(uhg)) @ sh1d.T
        Juu[0,0,e] -= tau
        Juu[-1,-1,e] += tau
        
        Juuhat[0,0,e] = -uhath[0,e] + tau
        Juuhat[-1,-1,e] = uhath[1,e] - tau
        
        # uhat equation
        
        Juhatq[0,0,e] = -1
        Juhatq[-1,-1,e] = 1
        
        Juhatu[0,0,e] = tau
        Juhatu[-1,-1,e] = -tau
        
        Juhatuhat[0,0,e] = -tau + (+uhath[0,e])
        Juhatuhat[-1,-1,e] = tau + (-uhath[1,e])
        
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

    delta_uhat_vec = -np.linalg.solve(H,r)
    
    delta_q = np.zeros((npl,nt))
    delta_u = np.zeros((npl,nt))
    delta_uhat = np.zeros((2,nt))
    
    for e in range(nt):
        
        delta_uhat[:,e] = -np.squeeze(delta_uhat_vec[e:e+2])
            
        aux_mat = np.block([[Jqq[:,:,e],Jqu[:,:,e]],[Juq[:,:,e],Juu[:,:,e]]])
        aux_rhs1 = np.vstack([Jquhat[:,:,e],Juuhat[:,:,e]])
        aux_rhs2 = np.vstack([-r1[:,e],-r2[:,e]])
        aux_rhs2 = np.squeeze(np.reshape(aux_rhs2,(2*npl,1)))
        
        rhs = aux_rhs2 - aux_rhs1 @ delta_uhat[:,e] 
        
        qu_vec = np.linalg.solve(aux_mat,rhs)
        
        delta_q[:,e] = qu_vec[:npl]
        delta_u[:,e] = qu_vec[npl:]
        
        qh[:,e] += delta_q[:,e]
        uh[:,e] += delta_u[:,e]
        
        uhath[:,e] += delta_uhat[:,e]
        
    return qh, uh, uhath, delta_u, r

def newton_method(master, mesh, dbc, param, uh, qh, uhath, source = None, extra_src = None, alpha = None, tol = 5e-14):
    
    norm_r = 1
    
    ite_counter = 0
        
    # fig, ax = plt.subplots()
    
    # plot_1D(mesh,uh,ax)
    
    while(norm_r> tol and not ite_counter > 2000):
        
        print(f"Newton iteration {ite_counter}")
        
        qh, uh, uhath, delta_u, r = hdg_solve1D(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
        
        # norm_delta_uh = np.sqrt(l2_norm_1D(mesh,delta_u))
        norm_r = residual_norm(mesh,r)
        
        
        # print("delta_u norm: ",norm_delta_uh)
        print("r norm: ", norm_r)
        
        ite_counter += 1
    
    assert(not norm_r > tol), "Could not reduce the residuals"
    
    return qh, uh, uhath

def DIRK_step(A, b, c, dt, t, dbc, master, mesh, param, uh, qh, uhath, source = None):

    bc0 = dbc(0,t+dt)
    bc1 = dbc(1,t+dt)
    
    uhath[0,0] = bc0
    uhath[-1,-1] = bc1
    
    nt = len(mesh.t)
    npl = len(mesh.dgnodes[:,0,0])    
    
    uhn1 = np.zeros((npl,nt))
    qhn1 = np.zeros((npl,nt))
    uhathn1 = np.zeros((2,nt))

    stages = A.shape[0]
    
    d = np.linalg.inv(A)
    
    qhs = np.zeros((npl,nt,stages))
    uhs = np.zeros((npl,nt,stages))
    uhaths = np.zeros((2,nt,stages))
    
    for s in range(stages):
        
        alpha = d[s,s]/dt
        
        extra_src = alpha*uh
        
        if s == 0:
            
            pass
        
        else:
            
            for j in range(s):
                
                extra_src -= d[s,j]/dt * (uhs[:,:,j] - uh)
            
        qhs[:,:,s], uhs[:,:,s], uhaths[:,:,s] = newton_method(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
            
        
    uhn1 = uh
    
    update = np.zeros((npl,nt))
    inner_sum = np.zeros((npl,nt))
    
    for i in range(stages):
        for j in range(stages):
            
            inner_sum += dt*b[i]*d[i,j] * (uhs[:,:,j] - uh)/dt
            
        update += inner_sum
        inner_sum = np.zeros((npl,nt))
        
    uhn1 += update
    
    # uhn1 = uhs[:,:,-1]
    
    sh1d = np.squeeze(master.sh1d[:,0,:])
    sh1dx = np.squeeze(master.sh1d[:,1,:])
    
    eps = param['eps']
    
    taud = eps
    
    H = np.zeros((nt+1,nt+1))
    r = np.zeros((nt+1,1))


    Jqq_inv = np.zeros((npl,npl,nt))
    Jquhat = np.zeros((npl,2,nt))
    rhs1 = np.zeros((npl,nt))
    
    for e in range(nt):
        
        dg = mesh.dgnodes[:,0,e]
        
        xxi = sh1dx.T @ dg
        dsdxi = np.sqrt(xxi**2)
        
        Jqq = sh1d @ np.diag(master.gw1d * dsdxi * 1/eps) @ sh1d.T
        
        Jquhat[0,0,e] = 1
        Jquhat[-1,-1,e] = -1
        
        uhn1g = sh1d.T @ uhn1[:,e]
        
        rhs1[:,e] = -sh1dx @ np.diag(master.gw1d) @ uhn1g
        
        Juhatq = np.zeros((2,npl))
        Juhatq[0,0] = -1
        Juhatq[-1,-1] = 1
        
        tauc = 1.5*abs(np.max(uhn1g))
        
        tau = taud + tauc
        
        Juhatuhat = np.zeros((2,2))
        Juhatuhat[0,0] = -tau
        Juhatuhat[-1,-1] = tau
        
        rhs2 = np.zeros(2)
        rhs2[0] = -tau*uhn1[0,e]
        rhs2[1] = tau*uhn1[-1,e]
        
        Jqq_inv[:,:,e] = np.linalg.inv(Jqq)
        
        H_k = Juhatuhat - Juhatq @ Jqq_inv[:,:,e] @ Jquhat[:,:,e]
        r_k = rhs2 - Juhatq @ Jqq_inv[:,:,e] @ rhs1[:,e]

        indices = [e,e+1]

        r_k = np.reshape(r_k,(2,1))
        
        H[e:e+2,e:e+2] += H_k
        r[e:e+2] += r_k
        
    # Apply Dirichlet BC
    
    H[0,:] = np.zeros(nt+1)
    H[0,0] = 1.
    r[0] = bc0
    
    H[-1,:] = np.zeros(nt+1)
    H[-1,-1] = 1.
    r[-1] = bc1
        
    uhathn1_vec = np.linalg.solve(H,r)
    
    for e in range(nt):
        
        uhathn1[:,e] = np.squeeze(uhathn1_vec[e:e+2])
        
        qhn1[:,e] = Jqq_inv[:,:,e] @ (rhs1[:,e] - Jquhat[:,:,e] @ uhathn1[:,e])
        
    return qhn1, uhn1, uhathn1
              
        
def BE_step(dt, t, dbc, master, mesh, param, uh, qh, uhath, uhist, source = None):
    
    bc0 = dbc(0,t+dt)
    bc1 = dbc(1,t+dt)
    
    uhath[0,0] = bc0
    uhath[-1,-1] = bc1
    
    alpha = 1/dt
    
    extra_src = alpha*uh
    
    qhn1, uhn1, uhathn1 = newton_method(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
    
    return qhn1, uhn1, uhathn1

def BDF2_step(dt, t, dbc, master, mesh, param, uh, qh, uhath, uhist, source = None):
    
    bc0 = dbc(0,t+dt)
    bc1 = dbc(1,t+dt)
    
    uhath[0,0] = bc0
    uhath[-1,-1] = bc1
    
    alpha0 = 3/2
    
    alpha1 = -2
    
    alpha2 = 1/2
    
    alpha = alpha0/dt
    
    extra_src = -alpha1/dt*uhist[:,:,0] - alpha2/dt*uhist[:,:,1]
    
    qhn1, uhn1, uhathn1 = newton_method(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
    
    return qhn1, uhn1, uhathn1


def BDF3_step(dt, t, dbc, master, mesh, param, uh, qh, uhath, uhist, source = None):
    
    bc0 = dbc(0,t+dt)
    bc1 = dbc(1,t+dt)
    
    uhath[0,0] = bc0
    uhath[-1,-1] = bc1
    
    alpha0 = 11/6
    
    alpha1 = -18/6
    
    alpha2 = 9/6
    
    alpha3 = -2/6 
    
    alpha = alpha0/dt
    
    extra_src = -alpha1/dt*uhist[:,:,0] - alpha2/dt*uhist[:,:,1] - alpha3/dt*uhist[:,:,2]
    
    qhn1, uhn1, uhathn1 = newton_method(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
    
    return qhn1, uhn1, uhathn1

def BDF4_step(dt, t, dbc, master, mesh, param, uh, qh, uhath, uhist, source = None):
    
    bc0 = dbc(0,t+dt)
    bc1 = dbc(1,t+dt)
    
    uhath[0,0] = bc0
    uhath[-1,-1] = bc1
    
    alpha0 = 25/12
    
    alpha1 = -48/12
    
    alpha2 = 36/12
    
    alpha3 = -16/12
    
    alpha4 = 3/12
    
    alpha = alpha0/dt
    
    extra_src = -alpha1/dt*uhist[:,:,0] - alpha2/dt*uhist[:,:,1] - alpha3/dt*uhist[:,:,2] - alpha4/dt*uhist[:,:,3]
    
    qhn1, uhn1, uhathn1 = newton_method(master, mesh, dbc, param, uh, qh, uhath, source, extra_src, alpha)
    
    return qhn1, uhn1, uhathn1
    
    
    
    
    
    