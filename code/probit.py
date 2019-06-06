import numpy as np
from scipy.stats import norm

# http://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html#Probit


def probit_loglikeobs(params, y, x):
    """Construct Log-likelihood contribution per individual of a probit model.
    
    Args:
        params (pd.Series): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables
        
    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution 
            per individual.
    
    See here for a reference: https://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html#Probit.loglikeobs
    
    """
    FLOAT_EPS = np.finfo(float).eps
    q = 2 * y - 1
    X = x
    return np.log(np.clip(norm.cdf(q * np.dot(X, params)), FLOAT_EPS, 1))


def probit_score_obs(params, y, x):
    """Construct Probit model Jacobian for each observation.

    Args:
        params (np.array):The parameters of the model
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables

    Returns:
        jac (np.array): 2d array with the derivative of the loglikelihood for each observation 
        evaluated at `params`.

    """
    FLOAT_EPS = np.finfo(float).eps
    y = y
    X = x
    XB = np.dot(X, params)
    q = 2 * y - 1
    # clip to get rid of invalid divide complaint
    L = q * norm.pdf(q * XB) / np.clip(norm.cdf(q * XB), FLOAT_EPS, 1 - FLOAT_EPS)
    return L[:, None] * X


def probit_hessian(params, y, x):
    """Construct Probit model Hessian matrix of the log-likelihood.

    Args:
        params (np.array): The parameters of the model
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables

    Returns: 
        hess (np.array): 2d array with the Hessian, second derivative of 
        loglikelihood function, evaluated at `params`.

    """
    X = x
    XB = np.dot(X, params)
    q = 2 * y - 1
    L = q * norm.pdf(q * XB) / norm.cdf(q * XB)
    return np.dot(-L * (L + XB) * X.T, X)


def probit_loglike(params, y, x):
    """Construct Log-likelihood of the Probit model.
    
    Args:
        params (np.array): The parameters of the model
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables

    Returns: 
        loglike (float): The log-likelihood function of the model evaluated 
        at `params`.

    """
    return probit_loglikeobs(params, y, x).sum()
