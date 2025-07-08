import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence as apnc
from scipy.stats import gaussian_kde
from scipy.linalg import inv
from scipy.io import loadmat

import warnings

def calc_SV(xa, xf):
      Nx, Ne = xa.shape
      sqrt_Ne = np.sqrt(1/(Ne - 1))
      xma = np.mean(xa, axis = 1)
      xmf = np.mean(xf, axis = 1)
      xpa, xpf = sqrt_Ne*(xa - xma[:, np.newaxis]), sqrt_Ne*(xf - xmf[:, np.newaxis])
      vals, vecs = np.linalg.eigh(np.matmul(xpf.T, xpf))
      val_sort = vals.argsort()[::-1]
      vals = vals[val_sort]
      vecs = vecs[:, val_sort]
      SVe = np.matmul(xpf, vecs)
      SVi = np.matmul(xpa, vecs)
      energy = np.sum(SVe*SVe, axis = 0)/np.sum(SVi*SVi, axis = 0)
      return SVi, SVe, energy, vals


def create_periodic(sigma, m, dx):
      if m % 2 == 0: #Even
            cx = m/2
            x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
      else: #Odd
            cx = np.floor(m/2)
            x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
      wlc = np.exp(-((dx*(x))**2)/(2*sigma*2))
      B = np.zeros((m, m))
      for i in range(m):
            B[i, :] = wlc[m - i:2*m - i]
      B = np.where(B < 0, 0, B)
      return B

def find_beta(sum_exp, Neff):
    #sum_exp is of size Ne
    Ne = sum_exp.shape[0]
    beta_max = np.max([1, 10*np.max(sum_exp)])
    w = np.exp(-sum_exp)
    ws = np.sum(w)
    if ws > 0:
        w = w/ws
        Neff_init = 1/sum(w**2)
    else:
        Neff_init = 1
    
    if Neff == 1:
        return

    if Neff_init < Neff or ws == 0:
        ks, ke = 1, beta_max
        tol = 1E-5
        #Start Bisection Method

        for i in range(1000):
            w = np.exp(-sum_exp/ks)
            w = w/np.sum(w)
            fks = Neff - 1/np.sum(w**2)
            if np.isnan(fks):
                fks = Neff-1
            
            w = np.exp(-sum_exp/ke)
            w = w/np.sum(w)
            fke = Neff - 1/np.sum(w**2)

            km = (ke + ks)/2
            w = np.exp(-sum_exp/km)
            w = w/np.sum(w)
            fkm = Neff - 1/np.sum(w**2)
            if np.isnan(fkm):
                fkm = Neff-1
            if (ke-ks)/2 < tol:
                break

            if fkm*fks > 0:
                ks = km
            else:
                ke = km
            
        beta = km
        w = np.exp(-sum_exp/beta)
        w = w/np.sum(w)
        Nf = 1/np.sum(w**2)
    else:
        beta = 1
    return beta



def get_reg(Nx, Ne, C, hw, Neff, res, beta_max):
    beta = np.zeros((Nx, ))
    res_ind = np.where(res > 0.0)[0]
    beta[res <= 0.0] = beta_max
    Ny, Ne = hw.shape
    #hw is Ny x Ne
    for i in res_ind:
        wo = 0
        loc_one = np.where(C[:, i] == 1)[0]
        loc_not_one = np.where(C[:, i] != 1)[0]
        dum = np.empty_like(hw)
        dum[loc_one, :] = np.log(Ne*hw[loc_one, :]) 
        dum[loc_not_one, :] = np.log((Ne*hw[loc_not_one, :] - 1)*C[loc_not_one, i, None] + 1)
        for y in range(Ny):
            wo = wo - dum[y, :]
            wo = wo - np.min(wo)
        beta[i] = find_beta(wo, Neff)
        if res[i] < 1/beta[i]:
            beta[i] = 1/res[i]
            res[i] = 0
        else:
            res[i] = res[i] - 1/beta[i]

        beta[i] = np.min([beta[i], beta_max])
    return beta, res


# glue prior and resampled particles together given posterior moments
# that we're seeking to match and a localization length scale
# merging is done to match the vanilla pf solution when no localization is happening
# and to match the prior when we're at a state very far away from the current obs

def sampling(x, w, Ne):

    # Sort sample
    b = np.argsort(x)
    
    # Apply deterministic sampling by taking value at every 1/Ne quantile
    cum_weight = np.concatenate(([0], np.cumsum(w[b])))
    
    offset = 0.0
    base = 1 / (Ne - offset) / 2
    
    ind = np.zeros(Ne, dtype=int)
    k = 1
    for n in range(Ne):
        frac = base + (n / (Ne - offset))
        while cum_weight[k] < frac:
            k += 1
        ind[n] = k - 1
    ind = b[ind]

    # Replace removed particles with duplicated particles
    ind2 = -999*np.ones(Ne, dtype=int)
    for n in range(Ne):
        if np.sum(ind == n) != 0:
            ind2[n] = n
            dum = np.where(ind == n)[0]
            ind = np.delete(ind, dum[0])
    

    ind0 = np.where(ind2 == -999)[0]
    ind2[ind0] = ind
    ind = ind2
    
    return ind


def gaussian_L(x, y, r):
    return np.exp(-(y - x)**2 / (2 * r)).item()


#TODO Rewrite to not have to rely on scipy
def kddm(x, xo, w):
    Ne = len(w)
    sig = (max(x) - min(x)) / 6
    npoints = 300
    
    xmin = min(min(xo), min(x))
    xmax = max(max(xo), max(x))
    
    xd = np.linspace(xmin, xmax, npoints)
    qf = np.zeros_like(x)
    cdfxa = np.zeros_like(xd)
    
    for n in range(Ne):
        qf += (1 + erf((x - x[n]) / (np.sqrt(2) * sig))) / (2 * Ne)
        cdfxa += w[n] * (1 + erf((xd - xo[n]) / (np.sqrt(2) * sig))) / 2
    
    interp_func = interp1d(cdfxa, xd, bounds_error=False, fill_value="extrapolate")
    xa = interp_func(qf)
    
    if np.var(xa) < 1e-8:
        warnings.warn("Low variance detected in xa")
    
    if np.isnan(qf).any():
        warnings.warn("NaN values detected in qf")
    
    return xa

def diff_map(data, Neig, knn, bw, eigmin, Ns, train_frac, keep_rows=[], klb=0.0, plotW=False):
      rng = np.random.default_rng(58)

      N, M = data.shape

      # Feature-wise scaling
      vt = data.copy()
      for i in range(M):
          denom = np.max(np.abs(vt[:, i]))
          if denom > 0:
              vt[:, i] /= denom
   
      # if doing knn approach
      # Compute pairwise distances
      dt_all = squareform(pdist(vt))
      srtd_idx = np.argsort(dt_all, axis=1)
      dt = np.take_along_axis(dt_all, srtd_idx[:, :knn], axis=1)
      nidx = srtd_idx[:, :knn]
   
      chosen_bw = bw
      # Build weight matrix
      temp_w = np.exp(-(dt**2) / chosen_bw)
      row_idx = np.repeat(np.arange(N), knn)
      W = csr_matrix((temp_w.ravel(), (row_idx, nidx.ravel())), shape=(N, N))
      W = (W.T).maximum(W)  # Symmetrize
      if klb > 0:
          W.data = np.where(W.data > klb, W.data, np.zeros_like(W.data))
   
      row_sum = np.array(W.sum(axis=1)).ravel()
   
      n_keep = len(keep_rows)
   
      # choose train_frac % of rows with the lowest row sum; keep track of which rows (samples) that is
      if n_keep == 0:
          n_keep = np.ceil(N * train_frac).astype('int')
          keep_rows = np.sort(np.argpartition(row_sum, n_keep-1)[:n_keep])

   
      row_sum_keep = row_sum[keep_rows]
      W_keep = W[np.ix_(keep_rows, keep_rows)]
      alpha_train = 1.0 / np.sqrt(row_sum_keep)
      # alpha_train = 1.0 / np.sqrt(row_sum)
      ld = diags(alpha_train)
   
      # Diffusion operator
      DO = ld @ W_keep @ ld
      # DO = ld @ W @ ld
      # print(DO.sum(axis=1))
      DO = (DO.T).maximum(DO)  # Ensure symmetr
      DO = DO + 1e-10 * np.eye(n_keep)

      # Eigen decomposition
      v0 = rng.uniform(0, 1, n_keep)
      try:
          eig_vals, eig_vecs = eigsh(DO, k=Neig + 1, sigma=1.0001, which="LM", v0=v0)
      except apnc as e:
          eig_vals = e.eigenvalues
          eig_vecs = e.eigenvectors
          print(f'Only found {len(eig_vals)} out of {Neig + 1} eigenvectors')

   
      eig_vals = np.real(eig_vals)
      eig_vecs = np.real(eig_vecs)
   
      eig_vecs = (eig_vecs.T[np.argsort(eig_vals)][::-1]).T
      eig_vals = eig_vals[np.argsort(eig_vals)][::-1]

   
      # Filter eigenvalues
      valid_indices = np.where(eig_vals > eigmin)[0]
      if len(valid_indices) == 1:
          valid_indices = np.array([0, 1])
   
      return eig_vecs[:, valid_indices], eig_vals[valid_indices], alpha_train, chosen_bw, keep_rows


def diff_map_ext_nystrom(Xnew, Xtrain, V, D, alphaTrain, chosenBw, knn, Ns):


      N, M = Xtrain.shape
      N2 = Xnew.shape[0]
      
      Ndims = np.ndim(Xnew)
      if Ndims == 1:
            Xnew = np.expand_dims(Xnew, axis=0)  # Add a new first dimension
            
      Xtrain_scaled = Xtrain.copy()
      Xnew_scaled = Xnew.copy()
      
      Xtrain_scaled[:, :Ns] = Xtrain_scaled[:, :Ns] / Ns
      Xnew_scaled[:, :Ns] = Xnew_scaled[:, :Ns] / Ns
            
            
      for j in range(M):
            denom = np.max(np.abs(Xtrain[:, j]))
            if denom > 0:
                  Xtrain_scaled[:, j] /= denom
                  Xnew_scaled[:, j] /= denom
                        
                        
      dist_mat = cdist(Xnew_scaled, Xtrain_scaled)
      sorted_idx = np.argsort(dist_mat, axis=1)
      knn = min(knn, N)
                        
      W_yi = np.zeros((N2, N))
      for iNew in range(N2):
            nbrs = sorted_idx[iNew, :knn]
            dist_sub = dist_mat[iNew, nbrs]
            W_yi[iNew, nbrs] = np.exp(-dist_sub**2 / chosenBw)
                              
      row_sum_new = W_yi @ (alphaTrain**2)
      alphaNew = 1.0 / np.sqrt(row_sum_new)
                              
      T = W_yi * alphaTrain.T
      
      T *= alphaNew
      # T *= alphaNew[:, np.newaxis]
      
      Vnew = T @ V
      Vnew /= D
      
      Vnew *= np.mean(alphaTrain)
                              
      return Vnew


def rkhs_likelihood(a, b, Neig, knn, klb, bw, Ns, train_frac):


      N = a.shape[0]
      a_cop = a.copy()
      b_cop = b.copy()

      # Get eigenvectors and eigenvalues of diffusion maps
      Vb, Db, a_train_b, bwb, keeps = diff_map(b_cop, Neig, knn, bw, 0.01, Ns, train_frac, plotW=True, klb=klb)
      Va, Da, a_train_a, bwa, keeps = diff_map(a_cop, Neig, knn, bw, 0.01, 1, train_frac, keeps, klb=klb)


      a_signs = np.where(Va[0] < 0, -1, 1)
      Va *= a_signs
      b_signs = np.where(Vb[0] < 0, -1, 1)
      Vb *= b_signs

      x_emb = (Vb * Db).T
      y_emb = (Va * Da).T

      a_cop = a_cop[keeps]
      b_cop = b_cop[keeps]

      for i in range(a_cop.shape[1]):
          # varb[i] = (np.sqrt(varb[i]) / np.max(np.abs(a[:, i]))) ** 2
          a_cop[:, i] /= np.max(np.abs(a_cop[:, i]))

      kde = gaussian_kde(a_cop.T, bw_method=bwa/a_cop.std(ddof=1))
      qa = kde(a_cop.T)

      # Calculate kernel mean embeddings
      N, M1 = Va.shape
      N, M2 = Vb.shape
      
      Va *= Da
      Vb *= Db

      Cab = (Va.T @ Vb) / N
      Cbb = (Vb.T @ Vb) / N

      C = Cab @ inv(Cbb, overwrite_a=True)
      Mu = (Vb @ C.T)
      
      # Compute pab
      pab = (Va @ Mu.T) * qa[:, np.newaxis]
      
      # Remove imaginary and negative values
      pab[pab < 0] = 0
      pab = np.real(pab)
      
      return pab, (Vb, Db, a_train_b, bwb), (Va, Da, a_train_a, bwa), keeps
