import numpy as np
import copy
from numbalsoda import lsoda_sig, solve_ivp, lsoda
from numba import cfunc
import numba as nb



def model(x, dt, T, funcptr):
      model_error = 0
      tspan = np.array([0, dt*T])
      usol = copy.deepcopy(x)
      #Try with a Runga Kutta Method first
      sol = solve_ivp(funcptr, tspan, usol, tspan, rtol = 1e-9, atol = 1e-30)
      tmp, success = sol.y, sol.success
      # There are points when L63 changes attractor when the problem becomes stiff 
      # If so, retry with a stiff LSODA solver
      if not success or np.allclose(tmp[-1, :], 0.0):
            tmp, success = lsoda(funcptr, usol, tspan, rtol = 1e-9, atol = 1e-30)
      if not success or np.allclose(tmp[-1, :], 0.0):
            model_error = 1
      return tmp[-1, :], model_error


def sigmoid(q, a=10, shift=0.0):
      return 1/(1+ np.exp(-a*(-q+shift)))

def fice_model(params={'a': 10, 'shift':0.0, 'h_noise':0.0, 'v_noise':0.0, 'Nx':40, 'bound':5}, dt=0, T = 0, funcptr=0):

      rng = np.random.default_rng(58)
      x_dom = np.linspace(-params['bound'], params['bound'], params['Nx']) 
      pre_threshold = sigmoid(x_dom, params['a'], params['shift'] + rng.normal(0, params['h_noise'])) 
      post_threshold = np.zeros_like(pre_threshold)
      for i in range(len(pre_threshold)):
            val = pre_threshold[i] + rng.normal(0,params['v_noise'])
            while val < 0 or val > 1:
                  val = pre_threshold[i] + rng.normal(0,params['v_noise'])
            post_threshold[i] = val

      return post_threshold

def linear_fice_model(params={'Nx':120, 'f': 1/3, 'v_noise':0.0}, dt=0, T=0, funcptr=0):
      rng = np.random.default_rng(58)
      Nx, f = params['Nx'], params['f']
      v_noise = params['v_noise']
      midpoint = Nx//2
      dec_start, dec_end = int(midpoint - Nx*f//2), int(midpoint + Nx*f//2)
      return np.concatenate((
                  np.ones(dec_start),
                  np.linspace(0, 1, dec_end - dec_start)[::-1],
                  np.zeros(Nx-dec_end),
            )) + rng.normal(0, v_noise, Nx)

def make_rhs_l63(kwargs):
      s = kwargs['s']
      r = kwargs['r']
      b = kwargs['b']      
      @cfunc(lsoda_sig)
      def rhs(t, u, du, p):
            du[0] = s*(u[1]-u[0])
            du[1] = u[0]*(r-u[2]) - u[1]
            du[2] = u[0]*u[1] - b*u[2]
      return rhs

def make_rhs_l96(kwargs):
      F = kwargs['F']
      Nx = 40
      @cfunc(lsoda_sig)
      def rhs(t, u, du, p):
            u_ = nb.carray(u, (Nx,))
            tmp = (np.roll(u_, -1) - np.roll(u_, 2))*np.roll(u_, 1) - u_ + F
            for i in range(Nx):
                  du[i] = tmp[i]
      return rhs

def make_rhs_l05(kwargs):
      K = int(kwargs['l05_K'])
      I = int(kwargs['l05_I'])
      b = kwargs['l05_b']
      c = kwargs['l05_c']
      F = kwargs['l05_F']
      Nx = 480
      K = np.round(K)
      I = np.round(I)
      alpha = (3*I**2 + 3)/(2*I**3 + 4*I)
      beta = (2*I**2+1)/(I**4 + 2*I**2)
      @cfunc(lsoda_sig)
      def rhs(t, z, dz, p):
            z_ = nb.carray(z, (Nx,))
            z0 = np.concatenate((z_, z_, z_))
            i = np.arange(-(I-1), I, dtype=np.int64)
            if I == 1:
                  x0 = z0
            else:
                  x0 = np.empty((Nx,))
                  for m in range(Nx):
                        n = Nx + m
                        x0[m] = np.sum((alpha - beta*np.abs(i))*z0[n+i]) + (alpha - beta*np.abs(-I))*z0[n-I]/2 + (alpha - beta*np.abs(I))*z0[n+I]/2
                  y0 = z0[Nx:2*Nx] - x0
            x0 = np.concatenate((x0, x0, x0))
            if I > 1:
                  y0 = np.concatenate((y0, y0, y0))
            w = np.empty((3*Nx))
            J = int(np.floor(K/2))
            j = np.arange(-(J-1), J, dtype = np.int64)
            if K%2 == 0:
                  norm = 1/2
            else:
                  norm = 1
            j = np.arange(-(J-1), J, dtype=np.int64)
            J = int(J)
            for m in np.arange(Nx-2*K, 2*Nx+2*K):
                   w[m] = (np.sum(x0[m-j]) + (x0[m-J] + x0[m+J])*norm)/K
            xx = np.empty((Nx,))
            for m in range(Nx):
                  n = Nx + m 
                  xx[m] = -w[n-2*K]*w[n-K] + (np.sum(w[n-K+j]*x0[n+K+j]) + (w[n-K-J]*x0[n+K-J] + w[n-K+J]*x0[n+K+J])*norm)/K
            i1 = Nx + np.arange(-2, Nx-2, dtype = np.int16)
            i2 = Nx + np.arange(-1, Nx-1, dtype = np.int16)
            i3 = Nx + np.arange(0, Nx, dtype = np.int16)
            i4 = Nx + np.arange(1, Nx+1, dtype = np.int16)

            if I>1:
                  yy = -y0[i1]*y0[i2] + y0[i2]*y0[i4]
                  yx = -y0[i1]*x0[i2] + y0[i2]*x0[i4]
                  tmp = xx + (b**2)*yy + c*yx - x0[i3] - b*y0[i3] + F
            else:
                  tmp = xx - x0[i3] + F
            for n in range(Nx):
                  dz[n] = tmp[n]
      return rhs

