import numpy as np
import copy
from . import MISC
import warnings
import functools
from scipy.io import loadmat
import matplotlib.pyplot as plt

#TODO Clean up and comment inside functions
#TODO Make the inflation calls in a separate function

def EnSRF_update(xf : np.ndarray, hx : np.ndarray, 
                 xm : np.ndarray, hxm: np.ndarray, 
                 y : np.ndarray, HC : np.ndarray, HCH: np.ndarray,
                 var_y : float, gamma : float, 
                 e_flag : int, qc : np.ndarray):
    '''Performs an Ensemble Square Root Filter update based on Whitaker and Hamill (2002).
    
    Parameters
    ----------
    xf : np.ndarray
        Array of size Nx x Ne containing the ensemble members
    hx : np.ndarray
        Array of size Ny x Ne containing the ensemble members projected into obs-space
    xm : np.ndarray
        Array of size Nx x 1 containinng the ensemble mean state
    hxm : np.ndarray
        Array of size Ny x 1 containing the ensemble mean projected into obs-space
    y : np.ndarray
        Array of size Ny x 1 containing the observations at time T
    HC : np.ndarray
        Localization matrix of size Ny x Nx in state-space
    HCH : np.ndarray
        Localization matrix of size Ny x Ny in obs-space
    var_y : float
        Observation variance
    gamma : float
        inflation parameter for RTPS
    e_flag : int
        Error flag
    qc : np.ndarray
        Array of length Ny repesenting quality control of observations

    Returns
    --------
    xa : np.ndarray
        Array of size Nx x Ne representing analysis ensemble states
    e_flag : int
        Error flag after data assimilation step
    '''
        
    if len(y.shape) == 1: #If Y is 1-dimensional, make it the correct dimension
        y = y[:, None]

    #Ensemble mean
    Ny = len(y[:, 0])
    Nx, Ne = xf.shape
    xp = xf - xm #Nx x Ne
    xpo = copy.deepcopy(xp) #Original perturbation
    hxp = hx - hxm # Ny x Ne
    #one obs at a time
    if e_flag !=0:
        return np.nan, e_flag
    if np.sum(qc) == Ny:
        warnings.warn('No observations pass QAQC, error_flag set to 1.')
        e_flag = 1
        return np.nan, e_flag

    for i in range(Ny):
        d = (y[i, :] - hxm[i, :])
        hxo = hxp[i, :]
        var_den = np.dot(hxo, hxo)/(Ne-1) + var_y
        P = np.dot(xp, hxo)/(Ne - 1)
        P = P*HC[i, :]
        K = P/var_den
        xm = xm + K[:, np.newaxis]*d[:, np.newaxis]

        beta = 1/(1 + np.sqrt(var_y/var_den))
        xp = xp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

        P = np.dot(hxp, hxo)/(Ne - 1)
        P = P*HCH[i, :]
        K = P/var_den

        hxm = hxm + K[:, np.newaxis]*d[:, np.newaxis]
        beta = 1/(1 + np.sqrt(var_y/var_den))
        hxp = hxp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

    #RTPS
    var_xpo = np.sqrt((1/(Ne-1))*np.sum(xpo*xpo, axis = 1)) #Nx x 1
    var_xp = np.sqrt((1/(Ne-1))*np.sum(xp*xp, axis = 1)) #Nx x 1
    inf_factor = gamma*((var_xpo-var_xp)/var_xp) + 1
    xp = xp*inf_factor[:, np.newaxis]
    return xm + xp, e_flag


def lpf_update(x : np.ndarray, hx : np.ndarray, 
               Y : np.ndarray, 
               H : np.ndarray, C_pf : np.ndarray, 
               N_eff : float, 
               min_res : int, maxiter : int, 
               kddm_flag : int,  
               e_flag : int, qcpass : np.ndarray,
               L: functools.partial, wc=100 ):

    '''Performs a Local Particle Filter update based on Poterjoy et al. (2022).

    
    Parameters
    ----------
    xf : np.ndarray
        Array of size Nx x Ne containing the ensemble members
    hx : np.ndarray
        Array of size Ny x Ne containing the ensemble members projected into obs-space
    Y : np.ndarray
        Array of size Ny x 1 containing the observations at time T
    H : np.ndarray
        Array of size Ny x Nx representing the measurement operator
    C_pf : np.ndarray
        Localization matrix of size Ny x Nx in state-space
    N_eff : float
        Effective Ensemble Size
    min_res : float
        Minimum residual for computing betas
    maxiter : int
        Maximum number of iterations of incremental LPF updates to perform
    kddm_flag : int
        Flag to turn on kernal density estimation. 0 for off, 1 for on. 
    e_flag : int
        Error flag
    qcpass : np.ndarray
        Array of length Ny repesenting quality control of observations

    Returns
    --------
    xa : np.ndarray
        Array of size Nx x Ne representing analysis ensemble states
    e_flag : int
        Error flag after data assimilation step
    '''
    if e_flag != 0 :
        return np.nan, e_flag

    if np.sum(qcpass) == len(Y):
        warnings.warn('No observations pass QAQC, error_flag set to 1.')
        e_flag = 1
        return np.nan, e_flag
    
    if len(Y.shape) == 1: #If Y is 1-dimensional, make it the correct dimension
        Y = Y[:, None]

    Nx, Ne = x.shape
    HCH = np.matmul(C_pf, H.T)

    # print(x.shape)
    # print(hx.shape)
    # print(Y.shape)
    # print(C_pf.shape)
    # print(HCH.shape)
    # print(N_eff)
    # print(min_res)
    # print(kddm_flag)
    # print(qcpass)
    # print(maxiter)

    Y = Y[qcpass == 0, :]
    hx = hx[qcpass == 0, :]
    C_pf = C_pf[qcpass == 0, :]
    HCH = HCH[qcpass == 0, :]
    HCH = HCH[:, qcpass == 0]
    Ny = len(Y)

    epsilon=1e-300
    max_res = 1.0
    beta = np.ones((Nx,))
    beta_y = np.ones((Ny,))
    
    res = np.ones(beta.shape)
    res_y = np.ones(beta_y.shape)
    niter = 0
    pf_infl = np.ones((Ny,))
    res_infl = np.ones(pf_infl.shape)

    res = res- min_res
    res_y = res_y - min_res

    #Beta stuff begins
    while max_res > 0 and min_res < 1:
        niter += 1
        xo = x.copy()
        hx = hx.squeeze()
        hxo = copy.deepcopy(hx)
        if len(hxo.shape) == 1:
             hxo = hxo[None, :]
        if len(hx.shape) == 1:
             hx = hx[None, :]
        
        omega = np.ones((Nx, Ne))*(1/Ne) #Nx x Ne
        omega_y = np.ones((Ny, Ne))*(1/Ne)
        lomega = np.zeros_like(omega)
        lomega_y = np.zeros_like(omega_y)


        wo = L(Y, hxo)
        wo = wo/np.sum(wo, axis = -1)[:, None]

        if np.any(np.isnan(wo)):
            e_flag = 1
            return np.nan, e_flag

        beta_y, res_y = MISC.get_reg2(Ny, Ne, HCH, wo, N_eff, res_y)
        beta, res = MISC.get_reg2(Nx, Ne, C_pf, wo, N_eff, res)
        wo_ind = np.where(1 < 0.99*Ne*np.sum(wo**2, axis = -1))[0]
        #Obs loop
        # for i in range(Ny):
        for i in wo_ind:

            beta_ind = np.where(beta != 0)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = C_pf[i, beta_ind] #Nxb array

            dum = np.zeros((len(beta_ind), Ne))

            # if np.any(C == 1.0):
            #     dum[C==1.0, :] = np.log(Ne*wo[i, :] + epsilon) 

            # dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1 + epsilon)

            val = C[:, None] * wt[None, :] + 1 + epsilon
            dum = np.log(val)


            lomega[beta_ind, :] = lomega[beta_ind, :] - dum

            lomega[beta_ind, :] = lomega[beta_ind, :] - np.nanmin(lomega[beta_ind, :], axis = -1)[:, None]
            lomega[beta_ind, :] = np.clip(lomega[beta_ind, :], None, wc)

            beta_ind = np.where(beta_y != 0)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = HCH[i, beta_ind] #Nxb array
    
            dum = np.zeros((len(beta_ind), Ne))
    
            # if np.any(C == 1.0):
            #     dum[C==1.0, :] = np.log(Ne*wo[i, :] + epsilon)
    
            # dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1 + epsilon)

            val = C[:, None] * wt[None, :] + 1
            dum = np.log(val)
    
            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - dum
            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - np.nanmin(lomega_y[beta_ind, :], axis = -1)[:, None]
            lomega_y[beta_ind, :] = np.clip(lomega_y[beta_ind, :], None, wc)

            #Normalize
            #lomega is Nx x Ne
            #omega needs to be Nx x Ne

            omega = np.exp(-lomega * beta[:, None])
            omega_y =  np.exp(-lomega_y * beta_y[:, None])


            omegas_y = np.sum(omega_y, axis = -1)[:, None] #Sum over Ensemble Members
            omegas = np.sum(omega, axis = -1)[:, None]

            omega = omega/omegas
            xmpf = np.sum(omega*xo, axis = -1)[:, None]
            omega_y = omega_y/ omegas_y
            hxmpf =np.sum(omega_y*hxo, axis = -1)[:, None]

            if (1 > 0.99*Ne*sum(wo[i, :]**2)):
                continue


            var_a = np.sum(omega*(xo - xmpf)**2, axis = -1)[:, None]
            var_a_y = np.sum(omega_y*(hxo - hxmpf)**2, axis = -1)[:, None]

            norm = (1 - np.sum(omega**2, axis = -1))[:, None]
            var_a = var_a/norm
            norm = (1 - np.sum(omega_y**2, axis = -1))[:, None]
            var_a_y = var_a_y/norm
            #ks = np.random.choice(Ne, Ne, p = omega_y[i, :], replace=True)

            
            ks = MISC.sampling(hxo[i, :], wo[i, :], Ne)

            x = _pf_merge(x, xo[:, ks], C_pf[i, :] * beta, Ne, xmpf, var_a)
            hx = _pf_merge(hx, hxo[:, ks], HCH[i, :] * beta_y, Ne, hxmpf, var_a_y)


        if kddm_flag == 1:
            for j in range(Nx):
                if np.var(x[j, :], ddof = 1) > 0:
                    # x[j, :] = MISC.kddm(x[j, :], xo[j, :], omega[j, :])
                    x[j, :] = MISC.kddm_fast(x[j, :], xo[j, :], omega[j, :])

            xmpf = np.mean(x, axis=1)

            for j in range(Ny):
                # hx[j, :] = MISC.kddm(hx[j, :], hxo[j, :], omega_y[j, :])
                hx[j, :] = MISC.kddm_fast(hx[j, :], hxo[j, :], omega_y[j, :])
        max_res = np.max(res_y)
        if niter == maxiter:
            break
    return x, e_flag

def lpf_update_keest_no_iter(x : np.ndarray, hx : np.ndarray, 
      Y : np.ndarray, 
      H : np.ndarray, C_pf : np.ndarray, 
      N_eff : float, wo: np.ndarray,
      min_res : int,  
      kddm_flag : int,  
      e_flag : int, wc=100):
    
      Nx, Ne = x.shape
      HCH = np.matmul(C_pf, H.T)

      Ny = len(Y)

      epsilon=1e-300
      beta = np.ones((Nx,))
      beta_y = np.ones((Ny,))
      
      res = np.ones(beta.shape)
      res_y = np.ones(beta_y.shape)
      
      res = res- min_res
      res_y = res_y - min_res
      
      xo = x.copy()
      hx = hx.squeeze()
      hxo = copy.deepcopy(hx)
      if len(hxo.shape) == 1:
            hxo = hxo[None, :]
      if len(hx.shape) == 1:
            hx = hx[None, :]

      omega = np.ones((Nx, Ne))*(1/Ne) #Nx x Ne
      omega_y = np.ones((Ny, Ne))*(1/Ne)
      lomega = np.zeros_like(omega)
      lomega_y = np.zeros_like(omega_y)


      
      if np.any(np.isnan(wo)):
            e_flag = 1
            return np.nan, e_flag

      beta_y, res_y = MISC.get_reg2(Ny, Ne, HCH, wo, N_eff, res_y)
      beta, res = MISC.get_reg2(Nx, Ne, C_pf, wo, N_eff, res)
      wo_ind = np.where(1 < 0.99*Ne*np.sum(wo**2, axis = -1))[0]

      #Obs loop
      for i in wo_ind:

            beta_ind = np.where(beta != 0)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = C_pf[i, beta_ind] #Nxb array

            dum = np.zeros((len(beta_ind), Ne))

            # if np.any(C == 1.0):
            #       dum[C==1.0, :] = np.log(Ne*wo[i, :] + epsilon) 

            # dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1 + epsilon)

            val = C[:, None] * wt[None, :] + 1 + epsilon
            dum = np.log(val)

            
            lomega[beta_ind, :] = lomega[beta_ind, :] - dum

            lomega[beta_ind, :] = lomega[beta_ind, :] - np.min(lomega[beta_ind, :], axis = -1)[:, None]
            lomega[beta_ind, :] = np.clip(lomega[beta_ind, :], None, wc)

            beta_ind = np.where(beta_y != 0)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = HCH[i, beta_ind] #Nxb array

            dum = np.zeros((len(beta_ind), Ne))

            # if np.any(C == 1.0):
            #       dum[C==1.0, :] = np.log(Ne*wo[i, :] + epsilon) 

            # dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1 + epsilon)

            val = C[:, None] * wt[None, :] + 1 + epsilon
            dum = np.log(val)

            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - dum
            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - np.min(lomega_y[beta_ind, :], axis = -1)[:, None]
            lomega_y[beta_ind, :] = np.clip(lomega_y[beta_ind, :], None, wc)

            #Normalize
            #lomega is Nx x Ne
            #omega needs to be Nx x Ne

            omega = np.exp(-lomega * beta[:, None])
            omega_y =  np.exp(-lomega_y * beta_y[:, None])


            omegas_y = np.sum(omega_y, axis = -1)[:, None] #Sum over Ensemble Members
            omegas = np.sum(omega, axis = -1)[:, None]

            omega = omega/omegas
            xmpf = np.sum(omega*xo, axis = -1)[:, None]
            omega_y = omega_y/ omegas_y
            hxmpf =np.sum(omega_y*hxo, axis = -1)[:, None]

            if (1 > 0.99*Ne*sum(omega_y[i, :]**2)):
                  continue

            var_a = np.sum(omega*(xo - xmpf)**2, axis = -1)[:, None]
            if len(hxmpf.shape) == 3:
                hxmpf = np.transpose(hxmpf, [0,2,1])
            var_a_y = np.sum(omega_y*(hxo - hxmpf)**2, axis = -1)[:, None]

            norm = (1 - np.sum(omega**2, axis = -1))[:, None]
            var_a = var_a/norm
            norm = (1 - np.sum(omega_y**2, axis = -1))[:, None]
            var_a_y = var_a_y/norm
            #ks = np.random.choice(Ne, Ne, p = omega_y[i, :], replace=True)
            if len(hxmpf.shape) == 3:
                ks = MISC.sampling(np.arange(len(wo[i, :])), wo[i, :], Ne)
            else:
                ks = MISC.sampling(hxo[i, :], wo[i, :], Ne)
            x = _pf_merge(x, xo[:, ks], C_pf[i, :] * beta, Ne, xmpf, var_a)
            if len(hxmpf.shape) == 3:
                hx = _pf_merge(hx, hxo[:, :, ks], HCH[i, :] * beta_y, Ne, hxmpf, var_a_y)
            else:
                hx = _pf_merge(hx, hxo[:, ks], HCH[i, :] * beta_y, Ne, hxmpf, var_a_y)

      if kddm_flag == 1:
            for j in range(Nx):
                  if np.var(x[j, :], ddof=1) > 0:
                        x[j, :] = MISC.kddm(x[j, :], xo[j, :], omega[j, :])

            xmpf = np.mean(x, axis=1)

            # for j in range(Ny):
            #       hx[j, :] = MISC.kddm(hx[j, :], hxo[j, :], omega_y[j, :])
      max_res = np.max(res)
      return x, e_flag


# def _pf_merge(x, xs, loc, Ne, xmpf, var_a):
#     '''Performs the merge step of the Local Particle Filter
    
#     Parameters
#     ------------
#     x : np.ndarray
    
#     xs : np.ndarray

#     loc : np.ndarray

#     Ne : int

#     xmpf : np.ndarray

#     var_a : ?

#     alpha : float

#     Returns
#     --------
#     xa : np.ndarray
#         Merged ensemble members
#     '''
#     if np.all(loc == 1):
#         xmpf = np.mean(xs, axis = -1)[:, None]
#         var_a = np.var(xs, axis = -1, ddof = 1)[:, None]
#     r1 = loc
#     r2 = 1-loc
#     xs = xs - xmpf
#     x = x - xmpf

#     #TODO 6/23 - updated code in the ELSE block to match JP updates to LPF - haven't yet tested the version for multivariate h(x) inside the IF block.
#     if(len(xs.shape)) == 3:
#         xa = xmpf + r1[:, None]*xs + r2[:, None]*x
#         pfm = (np.sum(xa, axis = -1)/Ne)[:, None]
#         pfm = np.transpose(pfm, [0,2,1])

#         pfv = (np.sum((xa - pfm)**2, axis=-1) / (Ne - 1))[:, None]

#         ratio = np.divide(var_a, pfv, out=np.ones_like(pfv, dtype=float), where=(pfv > 0))
#         xa = xmpf + (xa - pfm) * np.sqrt(ratio)

#     else: 
#         xa = xmpf + r1[:, None]*xs + r2[:, None]*x
#         pfm = (np.sum(xa, axis = -1)/Ne)[:, None]
#         pfv = (np.sum((xa - pfm)**2, axis=-1) / (Ne - 1))[:, None]
#         ratio = np.divide(var_a, pfv, out=np.ones_like(pfv, dtype=float), where=(pfv > 0))
#         xa = xmpf + (xa - pfm) * np.sqrt(ratio)

#     return xa

def _pf_merge(x, xs, loc, Ne, xmpf, var_a):
    '''Performs the merge step of the Local Particle Filter
    
    Parameters
    ------------
    x : np.ndarray
    xs : np.ndarray
    loc : np.ndarray
    Ne : int
    xmpf : np.ndarray
    var_a : np.ndarray
    alpha : float

    Returns
    --------
    xa : np.ndarray
        Merged ensemble members
    '''
    # print(var_a.shape)
    # Use keepdims=True to safely handle 2D and 3D shapes gracefully
    if np.all(loc == 1):
        xmpf = np.mean(xs, axis=-1, keepdims=True)
        var_a = np.var(xs, axis=-1, ddof=1, keepdims=True)
        
    r1 = loc
    r2 = 1 - loc
    xs = xs - xmpf
    x = x - xmpf

    xa = xmpf + r1[..., None] * xs + r2[..., None] * x
    pfm = np.sum(xa, axis=-1, keepdims=True) / Ne
    pfv = np.sum((xa - pfm)**2, axis=-1, keepdims=True) / (Ne - 1)
    
    if np.ndim(var_a) >= 2 and var_a.shape[-1] == var_a.shape[-2]:
        var_a = np.diagonal(var_a, axis1=-2, axis2=-1)
        
    if np.ndim(var_a) < np.ndim(pfv):
        var_a = var_a[..., None]
    out_shape = np.broadcast_shapes(np.shape(var_a), np.shape(pfv))
    
    ratio = np.divide(var_a, pfv, out=np.ones(out_shape, dtype=float), where=(pfv > 0))
    
    xa = xmpf + (xa - pfm) * np.sqrt(ratio)

    return xa
    

def lpf_update_keest_no_iter2(x : np.ndarray, hx : np.ndarray, 
      Y : np.ndarray, 
      H : np.ndarray, C_pf : np.ndarray, 
      N_eff : float, wo: np.ndarray,
      min_res : int,  
      kddm_flag : int,  
      e_flag : int, wc=100):
    
      Nx, Ne = x.shape
      Ny = len(Y)
      epsilon=1e-300
      beta = np.ones((Nx,))
      res = np.ones(beta.shape)
      res = res- min_res
      hx = hx.squeeze()
      hxo = copy.deepcopy(hx)
      xo = x.copy()

      omega = np.ones((Nx, Ne))*(1/Ne) #Nx x Ne
      lomega = np.zeros_like(omega)

      if np.any(np.isnan(wo)):
            e_flag = 1
            return np.nan, e_flag

      beta, res = MISC.get_reg2(Nx, Ne, C_pf, wo, N_eff, res)
      wo_ind = np.where(1 < 0.99*Ne*np.sum(wo**2, axis = -1))[0]

      #Obs loop
      for i in wo_ind:

            beta_ind = np.where(beta != 0)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = C_pf[i, beta_ind] #Nxb array

            dum = np.zeros((len(beta_ind), Ne))

            val = C[:, None] * wt[None, :] + 1 + epsilon
            dum = np.log(val)

            lomega[beta_ind, :] = lomega[beta_ind, :] - dum
            lomega[beta_ind, :] = lomega[beta_ind, :] - np.min(lomega[beta_ind, :], axis = -1)[:, None]
            lomega[beta_ind, :] = np.clip(lomega[beta_ind, :], None, wc)

            #Normalize
            #lomega is Nx x Ne
            #omega needs to be Nx x Ne

            omega = np.exp(-lomega * beta[:, None])
            omegas = np.sum(omega, axis = -1)[:, None]
            omega = omega/omegas
            xmpf = np.sum(omega*xo, axis = -1)[:, None]
            var_a = np.sum(omega*(xo - xmpf)**2, axis = -1)[:, None]
            norm = (1 - np.sum(omega**2, axis = -1))[:, None]
            var_a = var_a/norm
            #ks = np.random.choice(Ne, Ne, p = omega_y[i, :], replace=True)
            if 2 == 3:
                ks = MISC.sampling(hxo[:,i, :], wo[i, :], Ne)
            else:
                ks = MISC.sampling(hxo[i, :], wo[i, :], Ne)
            x = _pf_merge(x, xo[:, ks], C_pf[i, :] * beta, Ne, xmpf, var_a)

      if kddm_flag == 1:
            for j in range(Nx):
                  if np.var(x[j, :], ddof=1) > 0:
                        x[j, :] = MISC.kddm(x[j, :], xo[j, :], omega[j, :])

            xmpf = np.mean(x, axis=1)

      max_res = np.max(res)
      return x, e_flag




