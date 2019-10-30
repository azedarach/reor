"""
Provides test routines for FEM-BV-VARX routines.
"""

import unittest
import os
import numpy as np

from reor.fembv_varx import FEMBVVARXLocalLinearModel

TEST_DATA_PATH = os.path.realpath(os.path.dirname(__file__))


class TestFEMBVVARXLocalLinearModelFit(unittest.TestCase):
    """Provides unit tests for linear VARX models fitted with least squares."""

    def test_linear_varx_leastsq_example_1(self):
        """Test gives correct least-squares estimate for example dataset."""

        p = 2

        data = np.genfromtxt(os.path.join(TEST_DATA_PATH, 'test_dataset_1.csv'),
                             delimiter=',', names=True)

        # Use data from 1960Q1 to 1978Q4
        mask = data['year'] <= 1978

        invest = data['invest'][mask]
        income = data['income'][mask]
        cons = data['cons'][mask]

        # Fits are performed on first differences of log-data
        log_invest = np.log(invest)
        log_income = np.log(income)
        log_cons = np.log(cons)

        y = np.vstack(
            [np.diff(log_invest), np.diff(log_income), np.diff(log_cons)]).T

        self.assertTrue(y.shape == (75, 3))

        model = FEMBVVARXLocalLinearModel(y, p=p)
        model.fit()

        mu = model.mu
        A = model.A
        B = model.B
        B0 = model.B0
        Sigma_LS = model.Sigma_u

        expected_mu = np.array([-0.017, 0.016, 0.013])
        expected_A = np.array([[[-0.320, 0.146, 0.961],
                                [0.044, -0.153, 0.289],
                                [-0.002, 0.225, -0.264]],
                               [[-0.161, 0.115, 0.934],
                                [0.050, 0.019, -0.010],
                                [0.034, 0.355, -0.022]]])
        expected_Sigma_LS = np.array([[21.30e-4, 0.72e-4, 1.23e-4],
                                      [0.72e-4, 1.37e-4, 0.61e-4],
                                      [1.23e-4, 0.61e-4, 0.89e-4]])

        self.assertTrue(np.allclose(mu, expected_mu, atol=1e-3))
        self.assertTrue(np.allclose(A, expected_A, atol=1e-3))
        self.assertTrue(B is None)
        self.assertTrue(B0 is None)
        self.assertTrue(np.allclose(Sigma_LS, expected_Sigma_LS, atol=1e-5))
