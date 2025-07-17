# author HMS 06/2025
# used for setting what distributions are used to sample errors for and assimilate
# obs when using the local particle filter (or other nongaussian DA method)

# sampled errors are controlled by the used_obs_err and used_obs_err_params parameters
# in the experiment class. to sample obs using a different error distribution,
# all that needs to happen is a case needs to be added to sample_errors corresponding
# to the distribution of interest. the case number and parameters then are passed to the above
# parameters when constructing an Expt.

# to assimilate obs assuming a given likelihood, add the likelihood to get_likelihood.
# this involves defining the entire likelihood function including its parameters
# and then also adding a case to return the function with the provided parameters set.
# the corresponding parameters in the Expt class are prescribed_obs_err and
# prescribed_obs_err_params.

import numpy as np
from functools import partial

GAUSSIAN = 0
STATE_DEP_GAUSSIAN = 1
TOY_ICE_OBS = 2
CAUCHY = 3
UNIFORM_DONT_USE_ME = 4


def sample_errors(states, used_obs_error, params, rng):

    # GAUSSIAN = 0
    # STATE_DEP_GAUSSIAN = 1

    errors = -999 * np.zeros_like(states)

    match used_obs_error:
        case 0:
            try:
                mu, sigma = params["mu"], params["sigma"]
            except KeyError:
                raise KeyError(f'Parameters mu and sigma not provided in {params}')
            errors = rng.normal(mu, sigma, size=states.shape)
        case 1:
            try:
                mu1 = params["mu1"]
                mu2 = params["mu2"]
                sigma1 = params["sigma1"]
                sigma2 = params["sigma2"]
                threshold = params["threshold"]
            except KeyError:
                raise KeyError(f'Parameters mu1, sigma1, mu2, sigma2, and threshold not provided in {params}')

            errs1 = rng.normal(mu1, sigma1, states.shape)
            errs2 = rng.normal(mu2, sigma2, states.shape)

            errors = np.where(states < threshold, errs1, errs2)

        case 2:
            try:
                mu, sigma = params["mu"], params["sigma"]
                lnmu, lnsigma, lnscale = params["lnmu"], params["lnsigma"], params['lnscale']
            except KeyError:
                raise KeyError(f'Parameters mu, sigma, lnmu, lnsigma not provided in {params}')

            errs_low =  rng.lognormal(lnmu, lnsigma, states.shape)/lnscale
            errs_mid =  rng.normal(mu, sigma, states.shape)
            errs_high = -rng.lognormal(lnmu, lnsigma, states.shape)/lnscale

            errors = np.where(states < 0.1, errs_low, errs_mid)
            errors = np.where(states > 0.9, errs_high, errors)

            for i, row in enumerate(errors):
                for j, val in enumerate(row):
                    state = states[i,j]
                    thresh_high = 1 - state
                    thresh_low = -1*state
                    while val <= thresh_low or val >= thresh_high:
                        if state <= 0.1:
                            val = rng.lognormal(lnmu, lnsigma)/lnscale
                        elif state >= 0.9:
                            val = -rng.lognormal(lnmu, lnsigma)/lnscale
                        else:
                            val = rng.normal(mu, sigma)
                    errors[i,j] = val

    return errors

def get_likelihood(prescribed_obs_error, params):

    def gaussian_l(y, hx, mu, sigma):
        d = (y - hx - mu) ** 2 / (2 * sigma**2)
        d -= np.min(d, axis=-1)[:,None]
        return np.exp(-d)

    def state_dep_gaussian_l(y, hx, mu1, mu2, sigma1, sigma2, threshold):
        l_low = np.exp(-((y - hx - mu1) ** 2 / (2 * sigma1**2)))
        l_high = np.exp(-((y - hx - mu2) ** 2 / (2 * sigma2**2)))
        return np.where(hx < threshold, l_low, l_high)

    match prescribed_obs_error:
        case 0:
            try:
                return partial(gaussian_l, mu=params["mu"], sigma=params["sigma"])
            except KeyError:
                raise KeyError(f'Parameters mu and sigma not provided in {params}')
        case 1:
            try:
                return partial(
                    state_dep_gaussian_l,
                    mu1=params["mu1"],
                    sigma1=params["sigma1"],
                    mu2=params["mu2"],
                    sigma2=params["sigma2"],
                    threshold=params["threshold"],
                )
            except KeyError:
                raise KeyError(f'Parameters mu1, sigma1, mu2, sigma2, and threshold not provided in {params}')
