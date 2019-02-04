from scipy.optimize import minimize
from core.signal import LtstdbSignal
from scipy.sparse import diags
import numpy as np


class TvFilter:
    def __init__(self, delta=1, method='L-BFGS-B'):
        self.delta = delta
        self.method = method

    def filter(self, signal: LtstdbSignal):
        n = len(signal)
        D = diags([-1, 1], offsets=[0, 1], shape=(n - 1, n))
        xhat = np.zeros_like(signal)

        def obj_fun(xhat):
            Dxhat = D * xhat
            phi_tv = np.linalg.norm(Dxhat, ord=1)
            error = np.linalg.norm(xhat - signal, ord=2)
            cost = error + self.delta * phi_tv
            return cost

        result = minimize(obj_fun, x0=xhat, options={'disp': False}, method=self.method)
        return result
