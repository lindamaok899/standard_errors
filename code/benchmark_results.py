import pandas as pd
import numpy as np
import statsmodels.api as sm
import patsy
from standard_errors import cov_hessian, cov_jacobian, cov_sandwich
from statsmodels.base.model import GenericLikelihoodModel

# =======================================================================================
# Loading data and neccesary functions to get input data
# ------------------------------------------------------
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

spector_data = sm.datasets.spector.load_pandas().data
formula = "GRADE ~ GPA + TUCE + PSI"
y, x, params = binary_processing(formula, spector_data)
nobs = len(x)
# =======================================================================================
# specify model classes to get hessian & jacobian cov matrices
# ------------------------------------------------------------
# Probit class
class OurProbit(GenericLikelihoodModel):
    def nloglikeobs(self, *args, **kwargs):
        their_probit = sm.Probit(self.endog, self.exog)
        return -their_probit.loglikeobs(*args, **kwargs)

    def score_obs(self, *args, **kwargs):
        their_probit = sm.Probit(self.endog, self.exog)
        print("score was called.")
        return their_probit.score_obs(*args, *kwargs)

    def hessian(self, *args, **kwargs):
        their_probit = sm.Probit(self.endog, self.exog)
        print("hessian was called")
        return their_probit.hessian(*args, **kwargs)


# Logit class
class OurLogit(GenericLikelihoodModel):
    def nloglikeobs(self, *args, **kwargs):
        their_logit = sm.Logit(self.endog, self.exog)
        return -their_logit.loglikeobs(*args, **kwargs)

    def score_obs(self, *args, **kwargs):
        their_logit = sm.Logit(self.endog, self.exog)
        print("score was called.")
        return their_logit.score_obs(*args, *kwargs)

    def hessian(self, *args, **kwargs):
        their_logit = sm.Logit(self.endog, self.exog)
        print("hessian was called")
        return their_logit.hessian(*args, **kwargs)


# =======================================================================================
# estimate a probit model in statsmodels and compare cov_matrix results
# ---------------------------------------------------------------------
probit_mod = OurProbit(y, x)
probit_res = probit_mod.fit(start_params=params.to_numpy())

hessian_matrix = probit_res.hessv
jacobian_matrix = probit_mod.score_obs(probit_res.params)

# statsmodels covariance matrices
cov_jacobian_probit = probit_res.covjac
cov_hessian_probit = -np.linalg.inv(probit_res.hessv)
cov_sandwich_probit = probit_res.covjhj

# my covariance matrices
mycov_jacobian_probit = cov_jacobian(jacobian_matrix, nobs)
mycov_hessian_proibt = cov_hessian(hessian_matrix, nobs)
cov_sandwich_probit = cov_sandwich(jacobian_matrix, hessian_matrix, nobs)


# estimate logit model in statsmodels and compare cov_matrix results
# -------------------------------------------------------------------

logit_mod = OurLogit(y, x)
logit_res = logit_mod.fit(start_params=params.to_numpy())

hessian_matrix = logit_res.hessv
jacobian_matrix = logit_mod.score_obs(logit_res.params)

# statsmodels covariance matrices
cov_jacobian_logit = logit_res.covjac
cov_hessian_logit = -np.linalg.inv(logit_res.hessv)
cov_sandwich_logit = logit_res.covjhj

# my covariance matrices
mycov_jacobian_logit = cov_jacobian(jacobian_matrix, nobs)
mycov_hessian_logit = cov_hessian(hessian_matrix, nobs)
mycov_sandwich_logit = cov_sandwich(jacobian_matrix, hessian_matrix, nobs)

# =======================================================================================
