import pandas as pd
import numpy as np
import statsmodels as sm
import patsy
import statsmodels.discrete.discrete_model as smp
from probit import probit_score_obs, probit_hessian
from logit import logit_score_obs, logit_hessian
from standard_errors import cov_hessian, cov_jacobian, cov_sandwich

# ========================================================================================
# Loading data and neccesary functions to get input data
# ------------------------------------------------------

spector_data = sm.datasets.spector.load_pandas().data
formula = "GRADE ~ GPA + TUCE + PSI"


def binary_processing(formula, data):
    """Construct the inputs for the binomial functions.
    
    Args:
        formula (str): A patsy formula
        data (pd.DataFrame): The dataset
        
    Returns:
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables
        params_sr (pd.Series): The data are naive start values for the parameters.
            The index contains the parameter names (i.e. names of the variables).
    
        
    Patsy is described here: https://patsy.readthedocs.io/en/latest/
    
    Should we rather use DataFrames instead of numpy arrays?
    
    """
    y, x = patsy.dmatrices(formula, data, return_type="dataframe")
    start_params = np.linalg.lstsq(x, y, rcond=None)[0].ravel()
    params_sr = pd.Series(data=start_params, index=x.columns, name="value")
    return y.to_numpy().reshape(len(y)), x.to_numpy(), params_sr


# inputs
y, x, params = binary_processing(formula, spector_data)
nobs = len(x)
# =======================================================================================
# estimate a probit and logit model in statsmodels
# ------------------------------------------------

y, x, params_sr = binary_processing(formula, spector_data)
probit_mod = smp.Probit(endog=y, exog=x)
logit_mod = smp.Logit(endog=y, exog=x)

# fit probit and logit models
probit_res = probit_mod.fit()
logit_res = logit_mod.fit()

print(probit_res.summary())
print(logit_res.summary())
# =======================================================================================
# Estimate hessian and jacobian matrices from statsmodels and Laura's functions
# -------------------------------------------------------------------------------
t = smp.Probit.cov_params_func_l1()
# get statsmodels hessian and jacobian matrices
smprobit_hessian = probit_mod.hessian(params_sr)
smprobit_jacobian = probit_mod.score_obs(params_sr)

smlogit_hessian = logit_mod.hessian(params_sr)
smlogit_jacobian = logit_mod.score_obs(params_sr)

# get Lauras hessian and jacobian matrices
probit_hessian = probit_hessian(params, y, x)
logit_hessian = logit_hessian(params, y, x)

probit_jacobian = probit_score_obs(params, y, x)
logit_jacobian = logit_score_obs(params, y, x)

# =======================================================================================
# benchmark covariance matrices from Lauras's functions against statsmodels
# --------------------------------------------------------------------------

# compare probit and logit hessian cov matrices
cov_probit_hessian = cov_hessian(probit_hessian, nobs)
cov_probit_smhessian = cov_hessian(smprobit_hessian, nobs)

cov_logit_hessian = cov_hessian(logit_hessian, nobs)
cov_logit_smhessian = cov_hessian(smlogit_hessian, nobs)

# compare probit and logit jacobian cov matrices
cov_probit_jacobian = cov_jacobian(probit_jacobian, nobs)
cov_probit_smjacobian = cov_jacobian(smprobit_jacobian, nobs)

cov_logit_jacobian = cov_jacobian(logit_jacobian, nobs)
cov_logit_smjacobian = cov_jacobian(smlogit_jacobian, nobs)

# comapre probit and logit cov_sandwich results
covsand_probit_jacobian = cov_sandwich(probit_jacobian, probit_hessian, nobs)
covsand_probit_jacobiansm = cov_sandwich(smprobit_jacobian, smprobit_hessian, nobs)

covsand_logit_jacobian = cov_sandwich(logit_jacobian, logit_hessian, nobs)
covsand_logit_jacobiansm = cov_sandwich(smlogit_jacobian, smlogit_hessian, nobs)
# =======================================================================================
