'''Initialize, configure, and run data assimilation experiments with toy models.'''

__all__ = ['MISC', 'MODELS', 'DA']


import numpy as np
import multiprocessing as mp
import copy
import ast
from functools import partial
from . import MODELS
from . import MISC
from . import DA
from . import OBS_ERRORS
import xarray as xr
from . import Exceptions as dapExceptions
import pickle
import matplotlib.pyplot as plt
import warnings

from importlib import reload
reload(OBS_ERRORS)
reload(MISC)
reload(DA)

class Expt:
      '''Initialize a Data Assimilation experiment and configurable parameters.

      Attributes
      -----------
      modelParams: dict
            A dictionary containing all parameters initializing the chosen toy model.
      obsParams: dict
            A dictionary containing all parameters initializing observations.
      basicParams: dict
            A dictionary containing all parameters initializing basic experimental set-up.
      miscParams: dict
            A dictionary containing all parameters initializing output and other miscellaneous parameters.
      states: dict
            A dictionary containing all state vectors (model truth, observations, and initial ensemble states).

      '''

      def __init__(self, name : str, params : dict = None):
            #Expt Name
            self.exptname = name 

            #Dictionaries to store various experimental parameters
            self.modelParams= {}
            self.obsParams = {}
            self.basicParams = {}
            self.miscParams = {}
            self.states = {} #Dictionary to store all the model states

            if params is not None:
                  model_flag = params.get('model_flag')  
            else:
                  model_flag = None 
            #Initial the default parameters of an experiment
            self._initBasic(model_flag)
            self._initModel()
            self._initObs()
            self._initMisc()

            #If additional changes to the parameters are specified, change them
            if params is not None:
                  self.modExpt(params, reqUpdate=True)
            else:
            #If not, update and initalize basic ensemble member states
                  self._updateParams()

      def getParamNames(self) -> list:
            '''Retrieve all modifiable parameters in the Expt class.
            
            Parameters
            -----------
            None

            Returns
            -------
            list
                  a list of available parameters to modify
            '''
            paramList = []
            for d in [self. basicParams, self.obsParams, self.modelParams, self.miscParams]:
                  paramList.extend(list(d.keys()))
            paramList.remove('rhs')
            paramList.remove('funcptr')
            return paramList

      def __eq__(self, other):
            if not isinstance(other, Expt):
                  return False
            keys = self.getParamNames()
            equality = True
            for key in keys:
                  val1 = self.getParam(key)
                  val2 = other.getParam(key)
                  if isinstance(val1, np.ndarray):
                        equality = np.array_equal(val1, val2)
                  else:
                        equality = val1 == val2
                  if not equality:
                        return equality
            return equality


      def _spinup(self, Nx, Ne, dt, T, tau, funcptr, numPool, sig_y, h_flag, H):
            #Initial Ensemble
            #Spin Up
            seed = self.getParam('seed')
            if seed >= 0:
                  rng = np.random.default_rng(seed)
            else:
                  rng = np.random.default_rng()
            xt_0 = 3*np.sin(np.arange(Nx)/(6*2*np.pi))
            xt_0, model_error = MODELS.model(xt_0, dt, 100, funcptr)

            if model_error != 0:
                  warnings.warn('Model integration failed.')
                  self.modExpt({'status': 'init model error'})

            #Multiprocessing
            xf_0 = xt_0[:, np.newaxis] + 1*rng.standard_normal((Nx, Ne))
            pfunc = partial(MODELS.model, dt = dt, T = 100, funcptr=funcptr)
            
            with mp.get_context('fork').Pool(numPool) as pool:
                  pool_results  = pool.map(pfunc, [xf_0[:, i] for i in range(Ne)])
                  xf_0 = np.stack([x for x, _ in pool_results], axis = -1)
                  model_errors = np.array([y for _, y in pool_results])
            if np.any(model_errors != 0):
                  warnings.warn('Model integration failed.')
                  self.modExpt({'status': 'init model error'})
            #for n in range(Ne):
            #      xf_0[:, n], model_error = MODELS.model(xf_0[:, n], dt, 100, funcptr)
            #      if model_error != 0 :
            #           warnings.warn('Model integration failed.')
            #           self.modExpt({'status': 'init model error'})
      
            #Create Model Truth
            xt = np.zeros((Nx, T))
            xt[:,0], model_error = MODELS.model(xt_0, dt, 100, funcptr)

            if model_error != 0:
                  warnings.warn('Model integration failed.')
                  self.modExpt({'status': 'init model error'})

            for t in range(T-1):
                  xt[:, t+1], model_error = MODELS.model(xt[:, t], dt, tau, funcptr)
                  if model_error != 0:
                        warnings.warn('Model integration failed.')
                        self.modExpt({'status': 'init model error'})

            #Synthetic Observations
            used_obs_err = self.getParam('used_obs_err')
            used_obs_err_params = self.getParam('used_obs_err_params')

            match h_flag:
                  case 0:
                        Y_perf = np.matmul(H,xt)[:, :, np.newaxis]
                  case 1:
                        Y_perf = np.matmul(H, xt**2 )[:, :, np.newaxis] 
                  case 2:
                        Y_perf = np.matmul(H, np.log(np.abs(xt)))[:, :, np.newaxis] 
                  case _:
                        raise ValueError(f'Invalid Option Selected for Measurement Operator h: {h_flag}')

            Y = Y_perf + OBS_ERRORS.sample_errors(Y_perf, used_obs_err, used_obs_err_params, rng)

            return xf_0, xt, Y

      def _configModel(self):
            model_flag = self.getParam('model_flag')
            match model_flag:
                  case 0: #Lorenz 63
                        self.modelParams['rhs'] = MODELS.make_rhs_l63(self.modelParams['model_params'])
                        self.modelParams['Nx']  = 3
                  case 1: #Lorenz 96
                        self.modelParams['rhs'] = MODELS.make_rhs_l96(self.modelParams['model_params'])
                        self.modelParams['Nx'] = 40
                  case 2: #Lorenz 05
                        self.modelParams['rhs'] = MODELS.make_rhs_l05(self.modelParams['model_params'])
                        self.modelParams['Nx'] = 480
            self.modelParams['funcptr'] = self.modelParams['rhs'].address
      
      def _configObs(self):
            #Extra Observation stuff
            Nx = self.modelParams['Nx']
            H = np.eye(Nx) #Linear Measurement Operator
            H = H[self.obsParams['obb']:Nx-self.obsParams['obb']:self.obsParams['obf'], :]
            self.obsParams['H'] = H
            self.obsParams['Ny'] = len(H)
            #Create localization matrices
            if self.obsParams['localize']==1:
                  C = MISC.create_periodic(self.obsParams['roi'], Nx, 1/Nx)
                  #C_kf = MISC.create_periodic(self.obsParams['roi_kf'], Nx, 1/Nx)
                  #C_pf = MISC.create_periodic(self.obsParams['roi_pf'], Nx, 1/Nx)
            else:
                  C = np.ones((Nx, Nx))
                  #C_kf = np.ones((Nx, Nx))
                  #C_pf = np.ones((Nx, Nx))
            self.obsParams['C'] = np.matmul(H, C)
            #self.obsParams['C_kf'] = np.matmul(H, C_kf)
            #self.obsParams['C_pf'] = np.matmul(H, C_pf)

      def _clearAttributes(self):
            needed_attrs = ['exptname', 'modelParams', 'obsParams', 'basicParams', 'miscParams', 'states']
            all_attrs = list(self.__dict__.keys())
            for attr in np.setdiff1d(all_attrs, needed_attrs):
                  self.__delattr__(attr)

      def _updateParams(self):
            #Clear old RMSE, spread, and saved runtime attributes
            self._clearAttributes()
            #Model Truth            
            dt = self.basicParams['dt']
            Ne = self.basicParams['Ne']
            T = self.basicParams['T']
            tau = self.obsParams['tau']

            #Reset Error Flag
            self.basicParams['error_flag'] = 0
            
            self._configModel()

            self._configObs()

            Nx = self.getParam('Nx')
            h_flag = self.getParam('h_flag')
            H = self.getParam("H")
            #Do model spinup
            xf_0, xt, Y = self._spinup(Nx, Ne, dt, T, tau, self.getParam('funcptr'), self.getParam('NumPool'), self.getParam('sig_y'), h_flag, H)

            self.states['xf_0'] = xf_0
            self.states['xt'] = xt
            self.states['Y'] = Y


            #Initialize Variables for storage

            Ny = self.getParam('Ny')
            Nb = self.getParam('Nb')
            Ns = 1 + 2*Nb
            Nl = self.getParam('Nl')
            T_train = self.getParam('T_train')
            tof = self.getParam('tof')


            if self.getParam('saveEns') != 0:
                  self.x_ens = np.zeros((Nx, Ne, T))*np.nan #All Ensemble Members over time period
            if self.getParam('saveForecastEns') !=0:
                  self.x_fore_ens = np.zeros((Nx, Ne, T))
            if self.getParam('saveEnsMean') != 0:
                  self.x_ensmean = np.zeros((Nx, T))*np.nan
            if self.getParam('save_keest_pab') != 0:
                  self.keest_pab = np.zeros((T_train * Ny // tof, T_train * Ny // tof))*np.nan
                  self.x_train = np.zeros((Ns, T_train * Ny//tof))
                  self.y_train = np.zeros((Nl, T_train * Ny//tof))

            self.rmse = np.zeros((T,)) #RMSE of Expt
            self.rmse_prior = np.zeros((T,))
            self.spread = np.zeros((T,2)) #Spread of Expt Prio/Posterior

      #Modify the Experiment Parameters
      def _initBasic(self, model_flag = None):
            self.basicParams['T'] = 100
            match model_flag:
                  case 0: #L63
                        self.basicParams['dt'] = 0.01
                  case 1: #L96
                        self.basicParams['dt'] = 0.05
                  case 2: #L05
                        self.basicParams['dt'] = 0.05
                  case _: #None Case
                        self.basicParams['dt'] = 0.01
            self.basicParams['Ne'] = 10
            self.basicParams['expt_flag'] = 0
            self.basicParams['error_flag'] = 0
            self.basicParams['seed'] = -1
      def _initObs(self):
            self.obsParams['h_flag'] = 0 #Linear Operator
            self.obsParams['tau'] = 1     #Model steps between observations
            self.obsParams['obf'] = 1   #Observation spatial frequency: spacing between variables
            self.obsParams['obb'] = 0   #Observation buffer: number of variables to skip when generating obs
            #Localization
            self.obsParams['localize'] = 1
            self.obsParams['roi'] = 0.005
            #EnKF Parameters
            self.obsParams['inflation'] = 1
            self.obsParams['inf_flag'] = 0 #What Inflation Method to choose
            self.obsParams['gamma'] = 0.00

            #LPF Parameters
            self.obsParams['mixing_gamma'] = 0.3
            self.obsParams['kddm_flag'] = 0
            self.obsParams['min_res'] = 0.0
            self.obsParams['maxiter'] = 1
            self.obsParams['Nt_eff'] = 0.4

            #Observation Error Distribution Parameters
            default_gaussian_params = {'mu': 0, 'sigma' : 1}
            self.obsParams['used_obs_err'] = 0
            self.obsParams['used_obs_err_params'] = default_gaussian_params
            self.obsParams['prescribed_obs_err'] = 0
            self.obsParams['prescribed_obs_err_params'] = default_gaussian_params

            #Kernel Embeddings Likelihood Estimation Parameters
            # self.obsParams['do_keest'] = 0
            self.obsParams['nle_type'] = 2
            self.obsParams['Neig'] = 30
            self.obsParams['knn_frac'] = 0.05
            self.obsParams['bw_dm'] = 0.05
            self.obsParams['bw_kde'] = -999
            self.obsParams['T_train'] = 50
            self.obsParams['tof'] = 1
            self.obsParams['Nl'] = 1
            self.obsParams['Nb'] = 0

            #Parameters related to observation quality control
            self.obsParams['qc_flag'] = 0
      def _initModel(self):
            self.modelParams['model_flag'] = 0
            #Store the default parameters for all the possible models here
            params = {'s': 10, 'r': 28, 'b':8/3, 'F': 8, 
                      'l05_F':15, 'l05_Fe':15,
                      'l05_K':32, 'l05_I':12, 
                      'l05_b':10.0, 'l05_c':2.5}
            self.modelParams['model_params'] = params
      def _initMisc(self):
            #Output Parameters
            self.miscParams['status'] = 'init'
            self.miscParams['output_dir'] = './'
            #TODO Switch saveEns to off on release
            self.miscParams['saveEns'] = 1
            self.miscParams['saveEnsMean'] = 1
            self.miscParams['saveForecastEns'] = 0
            self.miscParams['save_keest_pab'] = 0


            #Default Singular Vector Parameters
            self.miscParams['doSV'] = 0 #0 for false, 1 for true
            self.miscParams['stepSV'] = 1 #how many timesteps to skip for each SV calculation
            self.miscParams['forecastSV'] = 4 # Optimization time interval for SV calculation
            self.miscParams['outputSV'] = self.getParam('output_dir') #output directory for SV calculation files
            self.miscParams['storeCovar']= 0 #Store the covariances 

            self.miscParams['numPool'] = 8

      def resetParams(self):
            '''Reset all Expt parameters to their default values.'''
            self.__init__(self.exptname)

      def modExptName(self, exptname : str) -> None:
            '''Modify the name of the Experiment

            Parameters
            -----------
            exptname : str
                  New experiment name

            '''
            self.exptname = exptname

      def modExpt(self, params : dict, reqUpdate: bool = False):
            '''Modify a parameter in the Expt class, recalculating new initial states if necessary.

            Note
            ------
            Modifying most parameters will cause a recalculation of all state vectors. To maintain consistent ensemble states across modification, consider setting the `seed` parameter in your experiment.

            Parameters
            -----------
            params : dict
                  A dictionary containing the parameter to modify and its new modified value
            reqUpdate : bool
                  A boolean determining whether recalculation of new states should be forced

            '''
            #CHECK Think about whether reseting experiment status should be done in _updateParams or here
            # Because modding the miscParams outDir technically doesn't change anything, but the status will get reset

            #Reset Experiment Status since a parameter has been modified
            self.miscParams['status'] = 'init'

            #Check if updating ensemble spinup required
            updateRequired = False
            for key, val in params.items():
                  if self.basicParams.get(key) is not None:
                        self.basicParams[key] = val
                        if key != 'expt_flag':
                              updateRequired = True
                  elif self.modelParams.get(key) is not None:
                        updateRequired = True
                        if key =='model_params':
                              model_params = self.modelParams[key]
                              for pkey, pval in val.items():
                                    model_params[pkey] = pval
                              self.modelParams['model_params'] = model_params
                        else:
                              self.modelParams[key] = val
                  elif self.obsParams.get(key) is not None:
                        updateRequired = True
                        self.obsParams[key] = val
                  elif self.miscParams.get(key) is not None:
                        self.miscParams[key] = val
                  else:
                        warnings.warn('({}, {}) key-value pair not in available configurable parameters. Ignoring.'.format(key, val))
            #Only update parameters if they affect model spinup
            if updateRequired or reqUpdate:
                  self._updateParams()

      def __str__(self):
             # Basic Model Setup Print
            ret_str = (
                  "------------------\n"
                  "Basic Information\n"
                  "------------------\n"
                  f"Experiment Name: {self.exptname}\n"
                  f"Ne: {self.basicParams['Ne']} # Number of Ensemble Members\n"
                  f"T: {self.basicParams['T']} # Number of Time Periods\n"
                  f"dt: {self.basicParams['dt']} # Width of Timesteps\n\n"
                  
                  "------------------\n"
                  "Model Information\n"
                  "------------------\n"
                  f"model_flag: {self.modelParams['model_flag']} # Model used in forward integration\n"
                  "      0: Lorenz 1963 (Nx = 3)\n"
                  "      1: Lorenz 1996 (Nx = 40)\n"
                  "      2: Lorenz 2005 (Nx  = 480)\n"
                  f"Nx: {self.modelParams['Nx']} # The number of state variables\n"
                  f"params: {self.modelParams['model_params']} # Parameters to tune each forecast model\n"
                  "Above is a list of all the parameters stored for use in the forecast model\n"
                  "      Lorenz 1963: [s, r, b]\n"
                  "      Lorenz 1996: [F]\n"
                  "      Lorenz 2005: [l05_F, l05_Fe, l05_K, l05_I, l05_b, l05_c]\n\n"
                  
                  "------------------------\n"
                  "Observation Information\n"
                  "------------------------\n"
                  f"h_flag: {self.obsParams['h_flag']} # Type of measurement operator to use\n"
                  "      0: Linear (x)\n"
                  "      1: Quadratic (x^2)\n"
                  "      2: Lognormal (log(abs(x)))\n"
                  f"used_obs_err: {self.obsParams['used_obs_err']} # todo \n"
                  f"used_obs_err_params: {self.obsParams['used_obs_err_params']} #todo \n"
                  f"prescribed_obs_err: {self.obsParams['prescribed_obs_err']} #todo \n"
                  f"prescribed_obs_err_params: {self.obsParams['prescribed_obs_err_params']} #todo \n"
                  f"tau: {self.obsParams['tau']} # Number of model time steps between data assimilation cycles\n"
                  f"obb: {self.obsParams['obb']} # Observation buffer: number of variables to skip when generating obs\n"
                  f"obf: {self.obsParams['obf']} # Observation spatial frequency: spacing between variables\n"
                  f"Ny: {self.obsParams['Ny']} # Number of observations to assimilate each cycle\n\n"
                  
                  "------------------------\n"
                  "DA Method Parameter Information\n"
                  "------------------------\n"
                  f"expt_flag: {self.basicParams['expt_flag']} # DA method for update step\n"
                  "      0: Ensemble Square Root Filter (EnSRF)\n"
                  "      1: Local Particle Filter (LPF)\n"
                  "      2: Stochastic EnKF \n"
                  "      3: No update (xa = xf)\n"
                  "      4: Kernel Embeddings Estimation LPF \n"
                  "      ...\n"
                  f"localize: {self.getParam('localize')} # Determines whether to apply localization\n"
                  "      0: Off\n"
                  "      1: On\n\n"
                  
                  "-----Kalman Filter (EnSRF)-----\n"
                  f"roi_kf: {self.getParam('roi_kf')} # Kalman Filter Localization Radius\n"
                  f"gamma: {self.getParam('gamma')} # RTPS parameter\n\n"
                  
                  "-----Local Particle Filter (LPF)-----\n"
                  f"roi_pf: {self.getParam('roi_pf')} # Particle Filter Localization Radius\n"
                  f"mixing_gamma: {self.getParam('mixing_gamma')} # Mixing coefficient for LPF\n"
                  f"kddm_flag: {self.getParam('kddm_flag')} # Determine whether to apply additional kernal density estimator in LPF step\n"
                  "      0: Off\n"
                  "      1: On\n"
                  f"maxiter: {self.getParam('maxiter')} # Maximum number of tempering iterations to run\n"
                  f"min_res: {self.getParam('min_res')} # Minimum residual\n"
                  f"Nt_eff: {self.getParam('Nt_eff')} # Effective Ensemble Size\n\n"
                  
                  "-----Kernel Embeddings Likelihood Estimation (KEEST)-----\n"
                  # f"do_keest: {self.getParam('do_keest')} # determine whether to estimate nonparametric likelihoods\n"
                  # "      0: Off\n"
                  # "      1: On\n"
                  f"nle_type: {self.getParam('nle_type')} # which flavor of KEEST to use.\n"
                  "      2: Estimating p(e|xt)\n"
                  "      3: Estimating p(e|xa)\n"
                  "      4: Estimating p(y|xt)\n"
                  "      5: Estimating p(y|xa)\n"
                  f"Neig: {self.getParam('Neig')} # number of eigenvectors to use for diffusion map embedding\n"
                  f"knn_frac: {self.getParam('knn_frac')} # fraction of nearest neighbors to use for diff maps\n"
                  f"bw_dm: {self.getParam('bw_dm')} # bandwidtch for diffusion maps kernel\n"
                  f"bw_kde: {self.getParam('bw_kde')} # bandwidth to use for KDE estimate of p(y); sentinel flag means adaptive selection\n"
                  f"T_train: {self.getParam('T_train')} # number of time steps of data to train on\n"
                  f"tof: {self.getParam('tof')} # training on every tof obs at each time\n"
                  f"Nl: {self.getParam('Nl')} # dimension of observation vectors\n"
                  f"Nb: {self.getParam('Nb')} # number of nearby states to use for model vector\n\n"
                  
                  "------------------------\n"
                  "Miscellaneous Information\n"
                  "------------------------\n"
                  f"status: {self.getParam('status')} # Notes the status of the given experiment\n"
                  "      init: The experiment has been initialized and spun-up, but not run using runDA\n"
                  "      init error: An error occured while spinning up the experiment\n"
                  "      run error: An error occured while running the experiment\n"
                  "      completed: runDA has been called and the experiment completed without errors\n"
                  f"output_dir: {self.getParam('output_dir')} # Default output dir for saving experiment-related material\n"
                  f"saveEns: {self.getParam('saveEns')} # Determines whether full posterior ensemble state is saved at each time step\n"
                  "      0: Off\n"
                  "      1: On (Default)\n"
                  f"saveEnsMean: {self.getParam('saveEnsMean')} # Determines whether ensemble mean is saved at each time step\n"
                  "      0: Off\n"
                  "      1: On (Default)\n"
                  f"saveForecastEns: {self.getParam('saveForecastEns')} #Determines whether full prior ensemble state is saved at each time step\n"
                  "      0: Off (Default)\n"
                  "      1: On\n\n"
                  f"save_keest_pab: {self.getParam('save_keest_pab')} # determines whether to save most recent estimate of p(a|b) from keest\n"
                  f"NumPool: {self.getParam('NumPool')} # Number of CPU cores to use when multiprocessing\n"
             )
            return ret_str

      def getBasicParams(self) -> tuple:
            '''Retrieve the basic parameters describing the experiment (Ne, Nx, T, and dt).
            
            Returns
            -------
            Ne : int
                  Number of ensemble members
            Nx : int
                  Number of model states
            T : int
                  Number of time steps the experiment will run for
            dt : float
                  The model time increment
            '''
            return self.basicParams['Ne'], self.modelParams['Nx'], self.basicParams['T'], self.basicParams['dt']
      
      def getStates(self) -> tuple:
            '''
            
            Returns
            -----------
            xf_0 : numpy.ndarray
                  Initial ensemble state that begins the experiment. Size (Nx, Ne)
            xt : numpy.ndarray
                  Model truth states throughout the experiment. Size (T, Nx)
            Y : numpy.ndarray
                  Observations throughout the experiment. Size (Ny, T, 1)
            '''
            return self.getParam('xf_0'), self.getParam('xt'), self.getParam('Y')
            #return self.states['xf_0'], self.states['xt'], self.states['Y']
      
      def getParam(self, param : str):
            '''Retrieve the value of a parameter stored in the Expt class.

            Note
            ------
            `getParam` returns **immutable** instances of the state vectors `xt`, `Y`, and `xf_0`, meaning that modifying instances returned by `getParam` will not modifying the instances stored in the `Expt` object itself. To modify the states of an experiment after spinup, access them directly through the `expt.states` dictionary.

            Parameters
            -----------
            param : str
                  Name of parameter to retrieve

            Returns
            -------
            param_value
                  Value of the requested parameter
            
            '''
            if self.basicParams.get(param) is not None:
                  val =  self.basicParams.get(param)
            elif self.modelParams.get(param) is not None:
                  val =  self.modelParams.get(param)
            elif self.obsParams.get(param) is not None:
                  val =  self.obsParams.get(param)
            elif self.modelParams['model_params'].get(param) is not None:
                  val =  self.modelParams['model_params'].get(param)
            elif self.states.get(param) is not None:
                  val = self.states.get(param)
            elif self.miscParams.get(param) is not None:
                  val =  self.miscParams.get(param)
            else:
                  return None
            #return copy.deepcopy(val)
            if isinstance(val, np.ndarray) or isinstance(val, dict):
                  return copy.deepcopy(val) #Make a deepcopy so that experiment values cannot be changed
            else:
                  return val
      
      def __deepcopy__(self, memo):
            expt = type(self)("", None)
            memo[id(self)] = expt
            #Change 'rhs' and 'funcptr' because 
            #they are incompatible with the deepycopy method
            self.modelParams['rhs'] = ''
            self.modelParams['funcptr'] = ''
            for name, attr in self.__dict__.items():
                  expt.__setattr__(name, copy.deepcopy(attr, memo))
            self._configModel()
            expt._configModel()
            return expt

      def __copy__(self):
            expt = type(self)("", None)
            expt.__dict__.update(self.__dict__)
            return expt
      

      def copyStates(self, expt):
            """Copy the states (model truth, initial ensemble, and observations) from another experiment

            Parameters
            ----------
            expt : Expt 
                The experiment to copy the states from.

            Raises
            ------
            TypeError
                Experiment instance not provided.
            dapExceptions.MismatchModelSize
                Model flags do not match between experiments.
            dapExceptions.MismatchTimeSteps
                Attempting to copy from experiment with less time steps than current experiment.
            dapExceptions.MismatchObs
                Number of assimilated observations do not match between experiments.
            dapExceptions.MismatchEnsSize
                Attempting to copy from experiment with less ensemble members than current experiment.
            dapExceptions.MismatchOperator
                Measurement operator flags do not match between experiments.
            """

            if not isinstance(expt, Expt):
                  raise TypeError("expt not an instance of the Expt class")
            
            Nx1, Nx2 = self.getParam('Nx'), expt.getParam('Nx')
            T1, T2 = self.getParam('T'), expt.getParam('T')
            obf1, obf2 = self.getParam('obf'), expt.getParam('obf')
            obb1, obb2 = self.getParam('obb'), expt.getParam('obb')
            Ne1, Ne2 = self.getParam('Ne'), expt.getParam('Ne')
            h_flag1, h_flag2 = self.getParam('h_flag'), expt.getParam('h_flag')
            if Nx1 != Nx2:
                  raise dapExceptions.MismatchModelSize(Nx1, Nx2)
            elif T1 > T2:
                  raise dapExceptions.MismatchTimeSteps(T1, T2)
            elif (obf1 != obf2) or (obb1 != obb2):
                  raise dapExceptions.MismatchObs(obf1, obf2)
            elif Ne1 > Ne2:
                  raise dapExceptions.MismatchEnsSize(Ne1, Ne2)
            elif h_flag1 != h_flag2:
                  raise dapExceptions.MismatchOperator(h_flag1, h_flag2)
            else:
                  if Ne1 < Ne2:
                        warnings.warn('Ensemble Sizes do not match between experiments. Taking subset of {}'.format(expt.exptname))
                  if T1 < T2:
                        warnings.warn('Time steps do not match between experiments. Taking subset of {}'.format(expt.exptname))
                  xf_0, xt, Y = expt.getStates()
                  self.states['xf_0'] = copy.deepcopy(xf_0[:, :Ne1])
                  self.states['xt'] = copy.deepcopy(xt)[:, :T1]
                  self.states['Y'] = copy.deepcopy(Y)[:, :T1, :]
                  
      def saveExpt(self, outputdir : str = None):
            """Save the experiment to the filesystem. Experiments are saved with the filename [exptname].expt

            Parameters
            ----------
            outputdir : str, optional
                Path to directory where to save the experiment , by default the value stored in the `output_dir` parameter.
            """
            #TODO Make sure that exptname is a valid filename, if not, convert it to one or throw an error
            if outputdir is None:
                  outputdir = self.getParam('output_dir')
            saveExpt(outputdir, self)

def loadParamFile(filename : str):
      """Create an experiment from params textfile.

      Parameters
      ----------
      filename : str
          Path to a textfile containing all configurable parameters, see sample files.

      Returns
      -------
      Expt
          An instance of the Expt class.
      """
      f = open(filename)
      lines = f.readlines()
      filtered = [x[:x.find('#')] for x in lines]
      expt_params = {}
      params = {}
      model_param_list = ['s', 'b', 'r', 'F', 
                          'l05_F', 'l05_Fe', 'l05_K', 'l05_I',
                          'l05_b', 'l05_c']
      for s in filtered:
            if s == '':
                  continue
            split = s.replace(' ', '').split('=')
            if split[0] == 'expt_name':
                  
                  expt_name = split[1].replace('\'', '').replace('\"', '')
            elif split[0] in model_param_list:
                  params[split[0]] = ast.literal_eval(split[1])
            else:
                  val = ast.literal_eval(split[1])
                  if val is str:
                        val = val.replace('\'', '').replace('\"', '')
                  expt_params[split[0]] = val
            expt_params['model_params'] = params
      f.close()
      e = Expt(expt_name, expt_params)
      return e

def saveExpt(outputdir : str, expt: Expt):
      """Save an experiment to the file system. Will save the file under 'exptname.expt'.

      Parameters
      ----------
      outputdir : str
          Path to directory where the experiment will be saved.
      expt : Expt
          The experiment instance to save.
      """
      expt.modelParams['rhs'] = ''
      expt.modelParams['funcptr'] = ''
      with open('{}/{}.expt'.format(outputdir, expt.exptname), 'wb') as f:
            pickle.dump(expt, f)
      expt._configModel()

def loadExpt(file : str):
      """Load an experiment from the filesystem.

      Parameters
      ----------
      file : str
          A string containing the filepath to the experiment.

      Returns
      -------
      Expt
          An instance of the Expt class.
      """
      with open(file, 'rb') as f:
            expt = pickle.load(f)
      expt._configModel()
      return expt

def plotLocalization(expt: Expt, ax = None):
      """Plot the localization radius of an experiment.

      Parameters
      ----------
      expt : Expt
          An Expt instance to plot localization for.
      ax : _type_, optional
          A matplotlib.pyplot `Axes` instance on which to plot localization onto, by default None. Must have a '3d' projection specified.

      Returns
      -------
      matplotlib.pyplot.Axes
          A `matplotlib.pyplot.Axes` instance.
      """
      Nx = expt.getParam('Nx')
      obb = expt.getParam('obb')
      C = expt.getParam('C')
      model_flag = expt.getParam('model_flag')
      if model_flag == 0: #L63 Case
            if ax is None:
                  fig, ax = plt.subplots(1, 1)
            xs = np.arange(Nx)
            ys = C[0, :]
            ys = np.roll(ys, -obb)
            ys = np.roll(ys, 1)
            ax.bar(xs, ys,  alpha = 0.6)
            ax.axvline(1, c = 'r',  label = r'$y_{o}$')
            ax.set_xticks(xs, labels = xs)
            ax.legend()
      elif model_flag == 1 or model_flag == 2: #L96 or L05 Case:
            if ax is None:
                  fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
            ts = np.linspace(0, 2*np.pi, Nx +1)
            xs, ys = np.cos(ts), np.sin(ts)
            zs = C[0, :]
            zs = np.roll(zs, -obb)
            zs = np.append(zs, zs[0])
            ax.plot(xs, ys, 0, c = 'k')
            ax.plot(xs, ys, 1, c = 'k')
            ax.fill_between(xs, ys, zs, xs, ys, 0, alpha = 0.6)
            ax.plot(xs[0], ys[0],[0, 1], c='r', label = r'$y_{o}$', zorder = 2)
            ax.view_init(elev=30, azim=25, roll=0)
            ax.legend()
      else: #All other potential future models
            pass
      if ax is None:
            return fig, ax

def plotExpt(expt: Expt, T: int, ax = None, plotObs = False, plotEns = True, plotEnsMean = False):
      """Plots the model truth, obs, ensembles, and ensemble mean at time T.

      Parameters
      ----------
      expt : Expt
          An Expt instance to plot
      T : int
          The time step during the experiment to plot.
      ax : matplotlib.pyplot.Axes, optional
          An instance of a matplotlib Axes to plot onto, by default None. Must have specified a '3d' projection
      plotObs : bool, optional
          A boolean to configure whether to plot observations, by default False
      plotEns : bool, optional
          A boolean to configure whether to plot ensemble members, by default True. Must have had saveEns configured to 1.
      plotEnsMean : bool, optional
          A boolean to configure whether to plot ensemble mean, by default False. Must have had saveEnsMean configured to 1.

      Returns
      -------
      matplotlib.pyplot.Axes
          A matplotlib.pyplot Axes instance

      Raises
      ------
      TypeError
          Raised if Expt instance is not passed.
      ValueError
          Raised if the Expt instance passed has not be run through `runDA()`, or an invalid time step, T, provided.

      """
      if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

      if not isinstance(expt, Expt):
            raise TypeError("expt not an instance of the Expt class")

      if expt.getParam('status') != 'completed':
            raise ValueError('Experiment has not run DA successfully')

            
      Nx = expt.getParam('Nx')
      Ne = expt.getParam('Ne')
      Nt = expt.getParam('T')
      H = expt.getParam('H')
      model_flag = expt.getParam('model_flag')

      if T < 0 or T > Nt:
            raise ValueError('Time step T ({}) not in range of Experiment ({})'.format(T, Nt))
      
      xf, xt, Y = expt.getStates()
      if plotEnsMean:
            x_ens = expt.x_ensmean[:, :T+1]
      if plotEns:
            x = expt.x_ens[:, :, :T+1]
      
      #Store all necessary information for plotting, depending on model
      if model_flag == 0:
            #Model Truth
            #Plot last 10 steps, if they exist
            diff = 10
            startT = T+1 - diff 
            if startT < 0:
                  startT = 0 
            xs_t, ys_t, zs_t = xt[0, startT:T+1], xt[1, startT:T+1], xt[2, startT:T+1]
            
            if plotEns:
                  xs, ys, zs = x[0, :, -diff:].T, x[1, :,  -diff:].T, x[2, :,  -diff:].T
            if plotEnsMean:
                  xs_ens, ys_ens, zs_ens = x_ens[0,  -diff:], x_ens[1,  -diff:], x_ens[2,  -diff:]
            if plotObs:
                  x_obs = copy.deepcopy(xt[:, T])
                  ind = np.array(np.where(H == 1))[1]
                  x_obs[ind] = Y[:, T, 0]
                  xs_y, ys_y, zs_y = x_obs[0], x_obs[1], x_obs[2]

      else:
            ts = np.linspace(0, 2*np.pi, Nx)
            xs, ys = np.cos(ts), np.sin(ts)

            #Ensemble
            if plotEns:
                  zs = x[:, :, -1]
            if plotEnsMean:
                  #Ensemble Mean
                  xs_ens, ys_ens = xs, ys 
                  zs_ens = x_ens[:, -1]

            #Obs
            if plotObs:
                  xs_y, ys_y = np.matmul(H, xs), np.matmul(H, ys)
                  zs_y = Y[:, T, 0]
            
            #Model Truth
            xs_t, ys_t = xs, ys
            zs_t = xt[:, T]
      
      #Start plotting on the axis object

      #Plot Ensemble Members
      if plotEns:
            for n in range(Ne):
                  if model_flag == 0:
                        ax.plot3D(xs[:, n], ys[:, n], zs[:, n], c = 'grey', alpha = 0.5)
                        ax.scatter(xs[-1, n], ys[-1, n], zs[-1, n], c = 'grey', alpha = 0.5)
                  else:
                        ax.plot3D(xs, ys, zs[:, n], c = 'grey', alpha = 0.5)

      #Plot Model Truth
      ax.plot3D(xs_t, ys_t, zs_t, 'blue', label = 'True')

      if model_flag==0:
            ax.scatter(xs_t[-1], ys_t[-1], zs_t[-1], c = 'blue')

      #Ensemble Mean
      if plotEnsMean:
            ax.plot3D(xs_ens, ys_ens, zs_ens, c = 'red', label = 'Post. Mean')
            if model_flag == 0:
                  ax.scatter(xs_ens[-1], ys_ens[-1], zs_ens[-1], c = 'red', label = 'Post. Mean')
      #Plot Observations
      if plotObs:
            if model_flag==0:
                  #Plot observations directly on 3d axis labels
                  #Adjust axis limits to fit points
                  old_xlim = ax.get_xlim()
                  old_ylim = ax.get_ylim()
                  old_zlim = ax.get_zlim()
                  xlim = (np.min([x_obs[0], old_xlim[0]]), np.max([x_obs[0], old_xlim[1]]))
                  ylim = (np.min([x_obs[1], old_ylim[0]]), np.max([x_obs[1], old_ylim[1]]))
                  zlim = (np.min([x_obs[2], old_zlim[0]]), np.max([x_obs[2], old_zlim[1]]))
                  for axis in ind:
                        match axis:
                              case 0: #X axis
                                    #Plot the model truth dot
                                    ax.scatter(xs_t[-1], ylim[0], zlim[0], c = 'blue')
                                    ax.scatter(x_obs[0], ylim[0], zlim[0], c = 'blue', marker = '^')
                              case 1: #Y axis
                                    ax.scatter(xlim[1], ys_t[-1], zlim[0], c = 'blue')
                                    ax.scatter(xlim[1], x_obs[1], zlim[0], c = 'blue', marker = '^')
                              case 2: #Z Axis
                                    ax.scatter(xlim[1], ylim[1], zs_t[-1], c = 'blue')
                                    ax.scatter(xlim[1], ylim[1], x_obs[2], c = 'blue', marker = '^')       
                  ax.set_xlim(xlim)
                  ax.set_ylim(ylim)
                  ax.set_zlim(zlim)

            else:
                  ax.scatter(xs_y, ys_y, zs_y, c = 'blue', marker = '^')

      #Return statement
      if ax is None:
            return fig, ax
      else:
            return ax

def plot_pab(expt: Expt, ax = None):

      if expt.getParam('save_keest_pab') == 0:
            raise ValueError('No likelihood estimate information saved. Be sure that save_keest_pab is set to true.')

      if ax is None:
            fig, ax = plt.subplots(1, 1)

      x_train = expt.x_train
      y_train = expt.y_train


      pab = expt.keest_pab
            
      
      ys = np.sort(y_train[0, :])
      ind2 = np.argsort(y_train[0, :])
      diff = np.diff(ys)

      fig = plt.figure(31)
      plt.clf()
      cmap = plt.get_cmap('tab20')

      htp = []

      for ind, j in enumerate(np.quantile(x_train, [0.1, 0.25, 0.5, 0.75, 0.9])):
            dum = np.abs(j - x_train[0, :])
            ind1 = np.argsort(dum)[0]

            h = pab[ind2, ind1]

            # Normalize for plotting
            normalization = np.sum((h[:-1] + h[1:]) * diff / 2)
            h_norm = h / normalization
            htp.append(h_norm)

            # Plotting
            ax.plot(ys, h_norm, linewidth=2, color=cmap(ind))
            ax.axvline(x=j, color=cmap(ind))

      bound = max(np.abs(np.min(x_train)), np.abs(np.max(x_train)))
      ax.set_xlim([-bound, bound])
      ax.set_ylim([0, np.max(htp)])

      ax.set_title(f'Conditional Likelihood Estimates for Experiment {expt.exptname}')

      if expt.getParam('nle_type') < 4:
            ax.set_xlabel('$\epsilon$')
            ax.set_ylabel('$p(\epsilon | x)$')
      else:
            ax.set_xlabel('$y$')
            ax.set_ylabel('$p(y | x)$')


      if ax is None:
            return fig, ax
      else:
            return ax

#TODO Add plotting funcitonality for Rank Histogram

def copyExpt(expt: Expt):
      """Create a deepcopy of an existing experiment

      Parameters
      ----------
      expt : Expt
          An instance of the Expt class.

      Returns
      -------
      Expt
          A copy of the experiment.
      """
      return copy.deepcopy(expt)


def runDA(expt: Expt, maxT : int = None):
      '''Run an experiment using its stored configurations
      
      Parameters
      ----------
      expt: Expt
            An instance of the Expt class you would like to run

      maxT : int, optional
            The maximum number of time steps to run the experiment out to.
            Must be less that the maximum time step "T" stored in the experiment. 
            If None, the experiment will run to the timestep specified by the "T" parameter.
      
      Returns
      ---------
      status : str
            A string specifying the status of the run.
      '''
      # Basic Parameters
      if expt.getParam('status') != 'init' and expt.getParam('status') != 'completed':
            raise ValueError('Experiment did not initialize successfully. Cannot run experiment further.')
      
      Ne, Nx, T, dt = expt.getBasicParams()

      if maxT is not None:
            if maxT > T:
                  raise ValueError('maxT greater than T in experiment ({} > {})'.format(maxT, T))
            T = maxT

      numPool = expt.getParam('numPool')
      #Observation Parameters
      H = expt.getParam('H')
      Ny = expt.getParam('Ny')
      tau = expt.getParam('tau')
      C = expt.getParam('C')
      Nt_eff = expt.getParam('Nt_eff')
      mixing_gamma = expt.getParam('mixing_gamma')
      min_res = expt.getParam('min_res')
      kddm_flag = expt.getParam('kddm_flag')
      maxiter = expt.getParam('maxiter')
      HC =  np.matmul(C,H.T)
      gamma = expt.getParam('gamma')
      prescribed_obs_err = expt.getParam('prescribed_obs_err')
      prescribed_obs_err_params = expt.getParam('prescribed_obs_err_params')

      # Nonparametric Likelihood Estimation Parameters
      nle_type = expt.getParam('nle_type')
      Neig = expt.getParam('Neig')
      knn_frac = expt.getParam('knn_frac')
      bw_dm = expt.getParam('bw_dm')
      bw_kde = expt.getParam('bw_kde')
      T_train = expt.getParam('T_train')
      tof = expt.getParam('tof')
      Nl = expt.getParam('Nl')
      Nb = expt.getParam('Nb')
      klb = 0.0
      train_frac = 1.0

      #Flags
      h_flag, expt_flag= expt.getParam('h_flag'), expt.getParam('expt_flag')
      qc_flag = expt.getParam('qc_flag')

      #Model Parameters
      params, funcptr = expt.getParam('model_params'), expt.getParam('funcptr')
      saveEns = expt.getParam('saveEns')
      saveEnsMean = expt.getParam('saveEnsMean')
      saveForecastEns = expt.getParam('saveForecastEns')

      e_flag = expt.getParam('error_flag')

      #Stored Statistics
      rmse = expt.rmse
      rmse_prior = expt.rmse_prior
      spread = expt.spread

      save_keest_pab = expt.getParam('save_keest_pab')
      if saveEns:
            x_ens = expt.x_ens
      if saveEnsMean:
            x_ensmean = expt.x_ensmean
      if saveForecastEns:
            x_fore_ens = expt.x_fore_ens

      # check if an observation error standard deviation was prescribed
      # this is necessary either if we are using a Kalman Filter variant
      # or if we are performing basic quality control on obs.
      if expt_flag == 0:
            if 'sigma' in prescribed_obs_err_params:
                  var_y = prescribed_obs_err_params['sigma']
            else:
                  raise KeyError(f'EnSRF Selected but no observation error standard deviation provided in prescribed_obs_err_params: {prescribed_obs_err_params}')

      if qc_flag == 1:
            if 'sigma' in prescribed_obs_err_params:
                  var_y = prescribed_obs_err_params['sigma']
            else:
                  raise KeyError(f'Obs QAQC turned on but no observation error standard deviation provided in prescribed_obs_err_params: {prescribed_obs_err_params}')
            if expt_flag == 3:
                  warnings.warn('QC Turned on but using nonparametric likelihood estiamation; QC will only be applied during training while using Gaussian likelihoods.')            

      #Open pool      
      #TODO Add exception handling in case function fails, 
      # make sure all the resources are released!

      rng = np.random.default_rng(58)

      pool = mp.get_context('fork').Pool(numPool)
      pfunc = partial(MODELS.model, dt = dt, T = tau, funcptr = funcptr)

      #Misc Parameters
      doSV = expt.getParam('doSV')
      #SV Parameters if flag is triggered
      if doSV==1:
            countSV = 0
            stepSV = expt.getParam('stepSV')
            forecastSV = expt.getParam('forecastSV')
            outputSV = expt.getParam('outputSV')
            storeCovar = expt.getParam('storeCovar')
                  #SV output variables
            xf_sv = np.zeros((Nx, Ne))
            sv_meta = {'expt_name': expt.exptname,
                  'T': T,
                  'stepSV': stepSV
                  }
            sv_coords = {'Nx': ('Nx', np.arange(Nx)),
                        'member': ('mem', np.arange(Ne)),
                        'time': ('t', np.arange(0, T, stepSV))}
            sv_t = len(sv_coords['time'][1])
            sv_data = {'initial': (['t', 'Nx', 'mem'], np.zeros((sv_t, Nx, Ne))*np.nan),
                  'evolved': (['t', 'Nx', 'mem'], np.zeros((sv_t, Nx, Ne))*np.nan),
                  'energy': (['t', 'mem'], np.zeros((sv_t, Ne))*np.nan),
                  'evalue': (['t', 'mem'], np.zeros((sv_t, Ne))*np.nan)}
            
            if storeCovar != 0:
                  sv_covar = {'Xa': (['t', 'Nx','mem'], np.zeros((sv_t, Nx, Ne))*np.nan), 
                                            'Xf': (['t', 'Nx','mem'], np.zeros((sv_t, Nx, Ne))*np.nan)}
            svpfunc = partial(MODELS.model, dt = dt, T = forecastSV, funcptr = funcptr)

      # set up bookkeeping stuff for kernel embeddings NLE
      knn = int(np.ceil(knn_frac * Ny * T_train))
      Ns = 1 + 2 * Nb
      x_train = np.zeros((Ns, T_train * Ny//tof))
      y_train = np.zeros((Nl, T_train * Ny//tof))

      if expt_flag == 3:
            Hi = np.zeros((Ny, Ns, Nx))

            for k in range(Ny):
                  ind = np.where(H[k,:] == 1)[0][0]
                  inds = np.arange(ind - Nb, k + Nb + 1)
                  inds[inds < 0] += Nx
                  inds[inds >= Nx] -= Nx
                  for l in range(Ns):
                        Hi[k, l, inds[l]] = 1

      # Time Loop
      xf_0, xt, Y = expt.getStates()
      xf = copy.deepcopy(xf_0)

      # Retrieve likelihood function (for use with LPF only)
      L = OBS_ERRORS.get_likelihood(prescribed_obs_err, prescribed_obs_err_params)

      for t in range(T):
            #Observation
            xm = np.mean(xf, axis = -1)[:, np.newaxis]
            rmse_prior[t] = np.sqrt(np.mean((xt[:, t] - xm[:, 0])**2))
            spread[t, 0] = np.sqrt(np.mean(np.sum((xf - xm)**2, axis = -1)/(Ne - 1)))
            if saveForecastEns:
                  x_fore_ens[:, :, t] = xf
            match h_flag:
                  case 0:
                        hx = np.matmul(H, xf)
                  case 1:
                        hx = np.matmul(H, np.square(xf))
                  case 2:
                        hx = np.matmul(H, np.log(np.abs(xf)))

            hxm = np.mean(hx, axis = -1)[:, None]
            qaqcpass = np.zeros((Ny,))

            #qaqc pass
            if qc_flag:

                  for i in range(Ny):
                        d = np.abs((Y[i, t, :] - hxm[i, :])[0])
                        if d > 4 * np.sqrt(np.var(hx[i, :]) + var_y):
                              qaqcpass[i] = 1
            #Data Assimilation
            #CHECK Make DA update steps return the ensemble mean, make LPF return xmpf specifically
            match expt_flag:
                  case 0: #Deterministic EnKF
                        xa, e_flag = DA.EnSRF_update(xf, hx, xm ,hxm, Y[:, t], C, HC, var_y, gamma, e_flag, qaqcpass)
                  case 1: #LPF
                        xa, e_flag = DA.lpf_update(xf, hx, Y[:, t], H, C, Nt_eff*Ne, mixing_gamma, min_res, maxiter, kddm_flag, e_flag, qaqcpass, L)
                  case 2: # Nothing
                        xa = xf
                  case 3: # LPF using kernel embeddings
                        if t < T_train:
                              xa, e_flag = DA.lpf_update(xf, hx, Y[:, t], H, C, Nt_eff*Ne, mixing_gamma, min_res, maxiter, kddm_flag, e_flag, qaqcpass, L)
                              ts = t * Ny // tof
                              te = ts + Ny // tof
                              s = 0 # sample # from this cycle

                              for k in range(0, Ny, tof):
                                    
                                    Htemp = Hi[k,:,:].squeeze()
                                    s = s+1
                                    # 2: p(e|xt)
                                    # 3: p(e|xa)
                                    # 4: p(y|xt)
                                    # 5: p(y|xa)
                                    if nle_type == 2 or nle_type == 4:
                                          x_train[:Ns,ts+s-1] = Htemp @ xt[:,t]
                                    elif nle_type == 3 or nle_type == 5:
                                          x_train[:Ns,ts+s-1] = Htemp @ rng.choice(xa)
                                          
                              if nle_type == 4 or nle_type == 5:
                                    y_train[ts:te] = Y[t,0::tof]
                              else:
                                    # HMS 6/11/25 - y is Ny x T x dummy ; note that this might get messed up when Nl > 0!
                                    y_train[:Nl,ts:te] = Y[0::tof,t,:].T - x_train[Nb,ts:te]
                        else:

                              print('about to start rkhs likelihood')
      
                              pab, x_map, y_map, keep_rows = MISC.rkhs_likelihood(y_train.T, x_train.T, Neig, knn, klb, bw_dm, Ns, train_frac)
                              print('out of rkhs likelihood')
                              if save_keest_pab != 0:
                                    expt.keest_pab = pab
                                    expt.x_train = x_train
                                    expt.y_train = y_train

                              pab += 1e-40

                              hxb_nbrs = np.zeros((Ns, Ne, Ny))

                              for n in range(Ne):
                                  # State
                                  for k in range(Ny):
                                      Htemp = np.squeeze(Hi[k, :, :])  # Equivalent to squeeze(Hi(k,:,:))
                                      if Ns == 1:
                                          Htemp = Htemp.T  # Transpose if Ns == 1
                                      hxb_nbrs[:, n, k] = Htemp @ xf[:, n]  # Matrix-vector multiplication
                              hxb_nbrs = np.transpose(hxb_nbrs, [0,2,1])


                              # compute particle weights outside of particle filter

                              Nxy = hxb_nbrs.shape[0]  # Equivalent to length(hx[:,0,0]) in MATLAB
                              evec_x, eval_x, a_train_x, bwx = x_map
                              evec_y, eval_y, a_train_y, bwy = y_map
                              x_emb = (evec_x * eval_x).T
                              y_emb = (evec_y * eval_y).T

                              wo = np.zeros((Ny, Ne))

                              for k in range(Ny):
                                    if nle_type >= 4 :
                                          obs_emb = MISC.diff_map_ext_nystrom(y[:,i-1,k].T,y_train.T,evec_y,eval_y,a_train_y,bwy,knn,1);
                                          obs_emb *= eval_y
                                          # ind2 = np.argmin(np.sum((Vyextra.T - Vyscaled.T) ** 2, axis=0))
                                    for n in range(Ne):
                                          if nle_type < 4:
                                                obs_emb = MISC.diff_map_ext_nystrom(
                                    # y_train[ts:te] = Y[t,0::tof]
                                                      (Y[k,t,:] - hxb_nbrs[(Nxy - 1) // 2, k, n]).T,
                                                      y_train.T,
                                                      evec_y,
                                                      eval_y,
                                                      a_train_y,
                                                      bwy,
                                                      knn,
                                                      1,
                                                )
                                                obs_emb *= eval_y  # Element-wise multiplication along columns
                                                ind2 = np.argmin(np.sum((obs_emb.T - y_emb) ** 2, axis=0))
                                          # Find nearest state on manifold
                                          state_emb = MISC.diff_map_ext_nystrom(
                                                hxb_nbrs[:, k, n].reshape(1, -1),
                                                x_train.T,
                                                evec_x,
                                                eval_x,
                                                a_train_x,
                                                bwx,
                                                knn,
                                                Ns,
                                          )
                                          state_emb *= eval_x  # Element-wise multiplication along columns
                                          ind1 = np.argmin(np.sum((state_emb.T - x_emb) ** 2, axis=0))
                                          wo[k,n] = pab[ind2, ind1]
                                    # sum of normalized likelihoods for a given observation equals one
                                    wo[k, :] /= np.sum(wo[k, :])

                              # placeholder - gaussian assimilation while i make sure that updating x_train and y_train works.

                              print('about to keest update')
                              xa, e_flag = DA.lpf_update_keest_no_iter(xf, hxb_nbrs, Y[:, t], H, C, Nt_eff*Ne, wo, mixing_gamma, min_res, kddm_flag, e_flag)
                              print('out of keest update')



                              # augment new data onto x_train and y_train
                              # TODO - play with looking at x train and y train before and after doing this and make sure you grok it all
                              l = 0
                              hxtemp = np.zeros((Ns, Ny // tof))

                              
                              for j in range(0, Ny, tof):
                                    l += 1
                                    Htemp = Hi[j, :, :].squeeze()
                                    if Ns == 1:
                                          Htemp = Htemp.T

                                    # Use truth or random sample from posterior
                                    if nle_type in [2, 4]:
                                          hxtemp[:, l - 1] = Htemp @ xt[:,t]
                                    else:
                                          dum = np.random.randint(0, Ne, size=1)
                                          hxtemp[:, l - 1] = hx[:, j, dum[0]]

                                  # Replace randomly-selected past data with new data
                              ind = np.arange(0,T_train*Ny)
                              rng.shuffle(ind)

                                    # y_train[:Nl,ts:te] = Y[0::tof,t,:].T - x_train[Nb,ts:te]

                              xtrinit = x_train.copy()
                              ytrinit = y_train.copy()
                              l = 0
                              for j in range(0, Ny, tof):
                                    l += 1
                                    x_train[:, ind[l - 1]] = hxtemp[:, l - 1]
                                    if nle_type in [4,5]:
                                          y_train[:Nl, ind[l - 1]] = Y[j, t,:]
                                    else:
                                          y_train[:Nl, ind[l - 1]] = Y[j, t, :] - hxtemp[Nb, l - 1]

                        #        populate the next entries of x_train and y_train
                        #        do normal LPF (see case 1)
                        # else, 
                        #        do estimation of p(whatever|whatever) (according to nle_type)
                        #        do LPF with keest-estimated likelihoods
                        #        replace a random element of the training dataset with this cycle's sample
                        

            if e_flag != 0:
                  pool.close()
                  expt.modExpt({'status': 'run DA error'})
                  print(expt.getParam('status'))
                  return expt.getParam('status')
            
            #TODO Add parameter that controls how often statistics are stored
            
            #Store the previous analysis into the matrix
            if saveEns:
                  x_ens[:, :, t] = xa
            if saveEnsMean:
                  x_ensmean[:, t] = np.mean(xa, axis = -1)
                 
            if doSV == 1 and t % stepSV == 0:
                  #Run SV calculation  
                  #xf_sv= np.stack(pool.map(svpfunc, [xa[:, i] for i in range(Ne)]), axis = -1)
                  pool_results  = pool.map(svpfunc, [xa[:, i] for i in range(Ne)])
                  xf_sv = np.stack([x for x, _ in pool_results], axis = -1)
                  model_errors = np.array([y for _, y in pool_results])
                  if np.any(model_errors != 0):
                        pool.close()
                        warnings.warn('Model integration failed at time T = {}. Terminating Experiment'.format(t))
                        expt.modExpt({'status': 'run model error'})
                        return expt.getParam('status')
                  (sv_data['initial'][1][countSV, :, :], sv_data['evolved'][1][countSV, :, :], 
                  sv_data['energy'][1][countSV, :], sv_data['evalue'][1][countSV, :]) = MISC.calc_SV(xa, xf_sv)
                  if storeCovar != 0:
                        sv_covar['Xa'][1][countSV, :, :] = xa
                        sv_covar['Xf'][1][countSV, :, :] = xf_sv
                  countSV+=1                  
            xma = np.mean(xa, axis = -1)
            rmse[t] = np.sqrt(np.mean((xt[:, t] - xma)**2))
            spread[t, 1] = np.sqrt(np.mean(np.sum((xa - xma[:, np.newaxis])**2, axis = -1)/(Ne - 1)))

            #Model integrate forward
            #Multiprocessing
            #xf = np.stack(pool.map(pfunc, [xa[:, i] for i in range(Ne)]), axis = -1)
            pool_results  = pool.map(pfunc, [xa[:, i] for i in range(Ne)])
            xf = np.stack([x for x, _ in pool_results], axis = -1)
            model_errors = np.array([y for _, y in pool_results])
            if np.any(model_errors != 0):
                  pool.close()
                  warnings.warn('Model integration failed at time T = {}. Terminating Experiment'.format(t))
                  expt.modExpt({'status': 'run model error'})
                  return expt.getParam('status')


            #No multiprocessing: Uncomment below for no multiprocessing
            #for n in range(Ne):
            #      tmp = copy.deepcopy(xa[:, n])
            #      xf[:, n], model_error = MODELS.model(tmp, dt, tau, funcptr)
            #      if model_error != 0:
            #           expt.modExpt({'status': 'run model error'})
            #           return expt.getParam('status')

      pool.close()
      # Save everything into a nice xarray format if SV calculations are on
      if doSV == 1:
            #Save everything into a netCDF here
            cdf = xr.Dataset(data_vars = sv_data, coords = sv_coords, attrs=sv_meta)
            if storeCovar != 0:
                  cdf = cdf.assign(sv_covar)
            cdf.to_netcdf('{}/SV_{}.cdf'.format(outputSV,expt.exptname), mode = 'w')

      #Output stuff
      expt.modExpt({'status': 'completed'})
      return expt.getParam('status')
