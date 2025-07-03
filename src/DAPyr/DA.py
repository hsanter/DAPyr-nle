import numpy as np
import copy
from . import MISC
import warnings
import functools

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
               N_eff : float, gamma : float, 
               min_res : int, maxiter : int, 
               kddm_flag : int,  
               e_flag : int, qcpass : np.ndarray,
               L: functools.partial ):
    '''Performs a Local Particle Filter update based on Poterjoy et al. (2022).
    
    Parameters
    ----------
    xf : np.ndarray
        Array of size Nx x Ne containing the ensemble members
    hx : np.ndarray
        Array of size Ny x Ne containing the ensemble members projected into obs-space
    Y : np.ndarray
        Array of size Ny x 1 containing the observations at time T
    var_y : float
        Observation variance
    H : np.ndarray
        Array of size Ny x Nx representing the measurement operator
    C_pf : np.ndarray
        Localization matrix of size Ny x Nx in state-space
    N_eff : float
        Effective Ensemble Size
    gamma : float
        Mixing coefficient parameter
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

    Y = Y[qcpass == 0, :]
    hx = hx[qcpass == 0, :]
    C_pf = C_pf[qcpass == 0, :]
    HCH = HCH[qcpass == 0, :]
    HCH = HCH[:, qcpass == 0]
    Ny = len(Y)

    max_res = 1.0
    beta = np.ones((Nx,))
    beta_y = np.ones((Ny,))
    beta_max = 1e100
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

        wo = L(Y, hxo) + 1E-40
        wo = wo/np.sum(wo, axis = -1)[:, None]

        if np.any(np.isnan(wo)):
            e_flag = 1
            return np.nan, e_flag

        beta_y, res_y = MISC.get_reg(Ny, Ne, HCH, wo, N_eff, res_y, beta_max)
        beta, res = MISC.get_reg(Nx, Ne, C_pf, wo, N_eff, res, beta_max)
        
        wo_ind = np.where(1 < 0.98*Ne*np.sum(wo**2, axis = -1))[0]
        #Obs loop
        for i in wo_ind:
            beta_ind = np.where(beta != beta_max)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = C_pf[i, beta_ind] #Nxb array
            dum = np.zeros((len(beta_ind), Ne))
            if np.any(C == 1.0):
                dum[C==1.0, :] = np.log(Ne*wo[i, :]) 
            dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1)
            lomega[beta_ind, :] = lomega[beta_ind, :] - dum
            lomega[beta_ind, :] = lomega[beta_ind, :] - np.min(lomega[beta_ind, :], axis = -1)[:, None]

            beta_ind = np.where(beta_y != beta_max)[0]
            wt = Ne*wo[i, :] - 1 #Ne Array
            C = HCH[i, beta_ind] #Nxb array
            dum = np.zeros((len(beta_ind), Ne))
            if np.any(C == 1.0):
                dum[C==1.0, :] = np.log(Ne*wo[i, :]) 
            dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1)
            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - dum
            lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - np.min(lomega_y[beta_ind, :], axis = -1)[:, None]

            #Normalize
            #lomega is Nx x Ne
            #omega needs to be Nx x Ne

            omega = np.exp(-lomega / beta[:, None])
            omega_y =  np.exp(-lomega_y / beta_y[:, None])

            omegas_y = np.sum(omega_y, axis = -1)[:, None] #Sum over Ensemble Members
            omegas = np.sum(omega, axis = -1)[:, None]

            omega = omega/omegas
            xmpf = np.sum(omega*xo, axis = -1)[:, None]
            omega_y = omega_y/ omegas_y
            hxmpf =np.sum(omega_y*hxo, axis = -1)[:, None]

            if (1 > 0.98*Ne*sum(omega_y[i, :]**2)):
                continue

            var_a = np.sum(omega*(xo - xmpf)**2, axis = -1)[:, None]
            var_a_y = np.sum(omega_y*(hxo - hxmpf)**2, axis = -1)[:, None]

            norm = (1 - np.sum(omega**2, axis = -1))[:, None]
            var_a = var_a/norm
            norm = (1 - np.sum(omega_y**2, axis = -1))[:, None]
            var_a_y = var_a_y/norm
            #ks = np.random.choice(Ne, Ne, p = omega_y[i, :], replace=True)
            ks = MISC.sampling(hxo[i, :], omega_y[i, :], Ne)
            x = _pf_merge(x, xo[:, ks], C_pf[i, :], Ne, xmpf, var_a, gamma)
            hx = _pf_merge(hx, hxo[:, ks], HCH[i, :], Ne, hxmpf, var_a_y, gamma)
        if kddm_flag == 1:
            for j in range(Nx):
                if np.var(x[j, :], ddof = 1) > 0:
                    x[j, :] = MISC.kddm(x[j, :], xo[j, :], omega[j, :])

            xmpf = np.mean(x, axis=1)

            for j in range(Ny):
                hx[j, :] = MISC.kddm(hx[j, :], hxo[j, :], omega_y[j, :])
        max_res = np.max(res)
        if niter == maxiter:
            break
    return x, e_flag

def _pf_merge(x, xs, loc, Ne, xmpf, var_a, alpha):
    '''Performs the merge step of the Local Particle Filter
    
    Parameters
    ------------
    x : np.ndarray
    
    xs : np.ndarray

    loc : np.ndarray

    Ne : int

    xmpf : np.ndarray

    var_a : ?

    alpha : float

    Returns
    --------
    xa : np.ndarray
        Merged ensemble members
    '''
    if np.all(loc == 1):
        xmpf = np.mean(xs, axis = -1)[:, None]
        var_a = np.var(xs, axis = -1, ddof = 1)[:, None]
    c = (1-loc)/loc
    xs = xs - xmpf
    x = x - xmpf

    var_a = var_a[:, 0]
    c2 = c**2
    v1 = np.sum(xs**2, axis = -1)
    v2 = np.sum(x**2, axis = -1)
    v3 = np.sum(x*xs, axis = -1)

    r1 = v1 + c2*v2 + 2*c*v3

    r2 = c2/r1

    r1 = alpha*np.sqrt((Ne-1)*var_a/r1)
    r2 = np.sqrt((Ne-1)*var_a*r2)

    if alpha < 1:
        m1 = np.mean(xs, axis = -1)
        m2 = np.mean(x, axis = -1)
        v1 = v1 - Ne*(m1**2)
        v2 = v2 - Ne*(m2**2)
        v3 = v3 - Ne*(m1*m2)
        T1 = v2
        T2 = 2*(r1*v3 + r2*v2)
        T3 = v1*(r1**2) + v2*(r2**2) + 2*v3*r1*r2 - (Ne-1)*var_a
        alpha2 = (-T2+np.sqrt((T2**2) - 4*T1*T3))/(2*T1)
        r2 = r2+alpha2
    
    xa = xmpf + r1[:, None]*xs + r2[:, None]*x
    pfm = (np.sum(xa, axis = -1)/Ne)[:, None]
    xa = xmpf + (xa - pfm)

    nanind = np.where(np.isnan(xa))
    xa[nanind] = xmpf[nanind[0], 0] + xs[nanind]

    return xa







