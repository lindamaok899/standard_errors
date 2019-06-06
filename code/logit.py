import numpy as np

# http://www.statsmodels.org/stable/_modules/statsmodels/discrete/discrete_model.html#Logit


def logit_loglikeobs(params, y, x):
    """Construct Log-likelihood contribution per individual of a logit model.
    
    Args:
        params (pd.Series**): The index consists of the parmater names,
            the values are the parameters.
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables
        
    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution 
            per individual.
            
    """
    q = 2 * y - 1
    X = x
    return np.log(1 / (1 + np.exp(-(q * np.dot(X, params)))))


def logit_score_obs(params, y, x):
    """Construct Logit model Jacobian of the log-likelihood for each observation.

    Args:
        params (np.array): The parameters of the model
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables

    Returns:
        jac (np.array): 2d array with the derivative of the loglikelihood for each 
        observation evaluated at `params`.
        
    """
    y = y
    X = x
    L = 1 / (1 + np.exp(-(np.dot(X, params))))
    return (y - L)[:, None] * X


def logit_hessian(params, y, x):
    """
    Construct Logit model Hessian matrix of the log-likelihood

    Args:
        params (np.array): The parameters of the model
        y (np.array): 1d numpy array with the dependent variable
        x_names (np.array): 2d numpy array with the independent variables

    Returns:
        hess (np.array): 2d array with the Hessian, second derivative of loglikelihood function,
        evaluated at `params'.

    """
    X = x
    L = 1 / (1 + np.exp(-(np.dot(X, params))))
    return -np.dot(L * (1 - L) * X.T, X)
