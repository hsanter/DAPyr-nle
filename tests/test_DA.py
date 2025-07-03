import sys
import unittest
sys.path.append("src")
import DAPyr as dap
import numpy as np
import pytest
import copy
import DAPyr.MISC
import DAPyr.Exceptions
import DAPyr.OBS_ERRORS

class TestDA(unittest.TestCase):

      def setUp(self):
            self.xf = np.loadtxt('./tests/states/xf.txt', delimiter = ',')
            self.Y = np.loadtxt('./tests/states/Y.txt', delimiter = ',')
            return super().setUp()
      
      def test_EnSRF_NoLocalize(self):
            xa_true = np.loadtxt('./tests/states/enkf_xa_nolocalize.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = self.xf.shape
            Ny = len(self.Y)
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            gamma = 0.0
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            var_y = 1
            xa, e_flag = dap.DA.EnSRF_update(xf, xf, xm , xm, Y[:, None], C, np.matmul(C, H.T), var_y, gamma, e_flag, qcpass)
            self.assertTrue(np.allclose(xa_true, xa))

      def test_EnSRF_Localize(self):
            xa_true = np.loadtxt('./tests/states/enkf_xa_localized_001.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = self.xf.shape
            Ny = len(self.Y)
            H = np.eye(Nx)
            roi = 0.001
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            gamma = 0.0
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            var_y = 1
            xa, e_flag = dap.DA.EnSRF_update(xf, xf, xm , xm, Y[:, None], C, np.matmul(C, H.T), var_y, gamma, e_flag, qcpass)
            self.assertTrue(np.allclose(xa_true, xa))
      @pytest.mark.filterwarnings("ignore:No observations")
      def test_EnSRF_qaqc(self):
            xf = self.xf
            Y = self.Y
            Nx, Ne = self.xf.shape
            Ny = len(self.Y)
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            gamma = 0.0
            e_flag = 0
            qcpass = np.ones((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            var_y = 1
            xa, e_flag = dap.DA.EnSRF_update(xf, xf, xm , xm, Y[:, None], C, np.matmul(C, H.T), var_y, gamma, e_flag, qcpass)
            self.assertTrue(np.isnan(xa))
            self.assertTrue(e_flag == 1)

      def test_EnSRF_e_flag(self):
            xf = self.xf
            Y = self.Y
            Nx, Ne = self.xf.shape
            Ny = len(self.Y)
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            gamma = 0.0
            e_flag = 1
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            var_y = 1
            xa, e_flag = dap.DA.EnSRF_update(xf, xf, xm , xm, Y[:, None], C, np.matmul(C, H.T), var_y, gamma, e_flag, qcpass)
            self.assertTrue(np.isnan(xa))
            self.assertTrue(e_flag == 1)

      def test_LPF_Localize(self):
            xa_true = np.loadtxt('./tests/states/lpf_xa_localize_001.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            roi = 0.001
            #C = np.ones((Nx, Nx))
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            prescribed_obs_error = 0
            prescribed_obs_error_params = {'mu': 0, 'sigma': 1}
            L = dap.OBS_ERRORS.get_likelihood(prescribed_obs_error, prescribed_obs_error_params)
            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)
            self.assertTrue(np.allclose(xa_true, xa))
      
      def test_LPF_NoLocalize(self):
            xa_true = np.loadtxt('./tests/states/lpf_xa_nolocalize.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            prescribed_obs_error = 0
            prescribed_obs_error_params = {'mu': 0, 'sigma': 1}
            L = dap.OBS_ERRORS.get_likelihood(prescribed_obs_error, prescribed_obs_error_params)
            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)
            print('marker')
            self.assertTrue(np.allclose(xa_true, xa))
      @pytest.mark.filterwarnings("ignore:No observations")
      def test_LPF_qaqc(self):
      #Test that if nothing passes QAQC, that an error flag is returned
            xf = self.xf
            Y = self.Y
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            roi = 0.001
            #C = np.ones((Nx, Nx))
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            e_flag = 0
            qcpass = np.ones((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            prescribed_obs_error = 0
            prescribed_obs_error_params = {'mu': 0, 'sigma': 1}
            L = dap.OBS_ERRORS.get_likelihood(prescribed_obs_error, prescribed_obs_error_params)
            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)
            self.assertTrue(np.isnan(xa))
            self.assertTrue(e_flag == 1)

      def test_LPF_e_flag(self):
            xf = self.xf
            Y = self.Y
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            roi = 0.001
            #C = np.ones((Nx, Nx))
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            e_flag = 1
            qcpass = np.zeros((Ny,))
            prescribed_obs_error = 0
            prescribed_obs_error_params = {'mu': 0, 'sigma': 1}
            L = dap.OBS_ERRORS.get_likelihood(prescribed_obs_error, prescribed_obs_error_params)
            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)
            self.assertTrue(np.isnan(xa))
            self.assertTrue(e_flag == 1)

      def test_Y_dim_LPF(self):
            xa_true = np.loadtxt('./tests/states/lpf_xa_localize_001.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            roi = 0.001
            #C = np.ones((Nx, Nx))
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            e_flag = 0
            qcpass = np.zeros((Ny,))
            prescribed_obs_error = 0
            prescribed_obs_error_params = {'mu': 0, 'sigma': 1}
            L = dap.OBS_ERRORS.get_likelihood(prescribed_obs_error, prescribed_obs_error_params)
            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            self.assertEqual(len(Y.shape), 1)
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)

            self.assertTrue(np.allclose(xa_true, xa))

      def test_Y_dim_EnSRF(self):
            xa_true = np.loadtxt('./tests/states/enkf_xa_localized_001.txt', delimiter = ',')
            xf = self.xf
            Y = self.Y
            Nx, Ne = self.xf.shape
            Ny = len(self.Y)
            H = np.eye(Nx)
            roi = 0.001
            C = dap.MISC.create_periodic(roi, Nx, 1/Nx)
            gamma = 0.0
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]
            var_y = 1
            self.assertEqual(len(Y.shape), 1)
            xa, e_flag = dap.DA.EnSRF_update(xf, xf, xm , xm, Y, C, np.matmul(C, H.T), var_y, gamma, e_flag, qcpass)
            self.assertTrue(np.allclose(xa_true, xa))


if __name__ == '__main__':
    unittest.main()
