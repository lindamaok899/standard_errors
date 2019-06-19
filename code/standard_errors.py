import numpy as np

# ======================================================================================
# Notes
# -----
## differences between hessian and jacobian covariance matrices

# in general the hessian and jacobian matrices will not have identical cov matrices.
# the hessian estimator typically has somewhat better properties in small samples.
# computation of the jacobian cov matrix requires the individual likelihood
# contributions

##Scaling

# scaling is not neccesary when using statsmodels and ose discrete models to estimate
# probit and logit hessian and jacobian covariance matrices as scaling results in very
# large standard errors.
# =======================================================================================


def cov_hessian(hessian):

    """Covariance based on the negative inverse of the hessian of loglike.
    
    Args:
        hessian (np.array): 2d array of dimension (nparams, nparams)
        nobs (scalar): number of observations
        
    Returns:
       hessian_matrix (np.array): 2d array covariance matrix (nparams, nparams)
       
   Notes
       - 
       - 
       - computation of the jacobian covariance matrix requires the individual likelihood 
       contributions.
     
   
    Resources: Marno Verbeek - A guide to modern econometrics.
    
    """

    info_matrix = -1 * (hessian)
    cov_hes = np.linalg.inv(info_matrix)

    return cov_hes


def cov_jacobian(jacobian):

    """Covariance based on outer product of jacobian of loglikeobs.
    
    Args:
        jacobian (np.array): 2d array of dimension (nobs, nparams)
        nobs (scalar): number of observations
        
    Returns:
        jacobian_matrix (np.array): 2d array covariance matrix (nparams, nparams)
        
     Notes:
       - in general the hessian and jacobian matrices will not have identical covariance
       matrices.
       - the hessian estimator typically has somewhat better properties in small samples.
       -computation of the jacobian covariance matrix requires requires the individual 
       likelihood contributions
        
    Resources: Marno Verbeek - A guide to modern econometrics.
    
    """

    info_matrix = np.dot((jacobian.T), jacobian)
    cov_jac = np.linalg.inv(info_matrix)

    return cov_jac


def cov_sandwich(jacobian, hessian):

    """Covariance of parameters based on HJJH dot product of Hessian, Jacobian, Jacobian, Hessian of likelihood.
    
    Args:
        jacobian (np.array): 2d array of dimension (nobs, nparams)
        nobs (scalar): number of observations
        
    Returns:
        sandwich_cov (np.array): 2d array covariance HJJH matrix (nparams, nparams)
        
    Resources:
    (https://github.com/statsmodels/statsmodels/blob/a33424eed4cacbeacc737d40fe5700daf39463f6/statsmodels/base/model.py#L2194)
    
    """

    info_matrix = np.dot((jacobian.T), jacobian)
    cov_hes = cov_hessian(hessian)
    sandwich_cov = np.dot(cov_hes, np.dot(info_matrix, cov_hes))

    return sandwich_cov


def se_calculations(function):

    """standard deviation of parameter estimates based on the function of choice.
    
    Returns:
        standard_errors (np.array): 1d array of dimension (nparams) with standard errors.
    
    """
    standard_errors = np.sqrt(np.diag(function()))

    return standard_errors
