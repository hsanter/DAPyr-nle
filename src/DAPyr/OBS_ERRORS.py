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
LOGNORMAL = 2
CAUCHY = 3
LOGISTIC = 5
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
        case 3:
            try:
                x0, gamma = params["x0"], params["gamma"]
            except KeyError:
                raise KeyError(f'Parameters x0 and gamma not provided in {params}')
            errors = x0 + gamma * rng.standard_cauchy(size=states.shape)
        case 5:
            try:
                x0, scale = params["x0"], params["scale"]
            except KeyError:
                raise KeyError(f'Parameters x0 and scale not provided in {params}')
            errors = rng.logistic(loc = x0, scale=scale, size=states.shape)

    return errors

def get_likelihood(prescribed_obs_error, params):

    def gaussian_l(y, hx, mu, sigma):
        d = (y - hx - mu) ** 2 / (2 * sigma**2)
        d -= np.min(d, axis=-1)[:,None]
        return np.exp(-d)

    def cauchy_l(y, hx, x0, gamma):
        square_me = (y - hx - x0)/gamma
        return 1/(np.pi * gamma * (1 + square_me**2))

    def logistic_l(y, hx, x0, scale):
        exp = np.exp(-(y - hx - x0)/scale)
        return exp/(scale * (1 + exp)**2)

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

        case 3:
            try:
                return partial(
                    cauchy_l,
                    x0=params["x0"],
                    gamma=params["gamma"],
                )
            except KeyError:
                raise KeyError(f'Parameters x0, gamma not provided in {params}')

        case 5:
            try:
                return partial(
                    logistic_l,
                    x0=params["x0"],
                    scale=params["scale"],
                )
            except KeyError:
                raise KeyError(f'Parameters x0, scale not provided in {params}')
