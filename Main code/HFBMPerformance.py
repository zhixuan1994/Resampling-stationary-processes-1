from fractal_analysis.tester.series_tester import FBMSeriesTester
from fractal_analysis.tester.critical_surface import CriticalSurfaceFBM
import numpy as np
import pandas as pd

# Input path with max lags (use to estimate Hurst exponent), 
# and estimated H (used to check whether the input path is FBM)
# Example: given path p, max_lag (default 40) and H0 (default 0):
# H extimate = fbmHcheck(p, max_lag).H_est; 
# is fbm = fbmHcheck(p, max_lag, H0).fbm_test.

# Reference: https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
class fbmHcheck:
    def __init__(self, path, max_lag=40, H=0):
        self.path = path
        self.max_lag = max_lag
        self.H = H
    
#     Calculate Hurst exponent by using variance of the lagged difference
    def H_est(self):
        time_series = self.path if not isinstance(self.path, pd.Series) else self.path.to_list()
        lags = range(2, self.max_lag)
        tau = []
        for lag in lags:
            tau.append(np.std(np.subtract(time_series[lag:], time_series[:-lag])))
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
        
#     Check whether the input path is Fractional Brownian motion
    def fbm_test(self):
        H = self.H_est()
        fbm_tester = FBMSeriesTester(critical_surface=CriticalSurfaceFBM(N=len(self.path), 
                                                                     alpha=0.05, 
                                                                     is_increment_series=False))
        is_fbm, sig2 = fbm_tester.test(h=self.H, x=self.path, sig2=None, add_on_sig2=0)
        return int(is_fbm)
