from __future__ import division
from mLGHOO import *
import logging
import scipy.optimize
import scipy.spatial
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)



class otherAlg():
    def __init__(self, func):
        self.func = func
        return

    def run(self, x0, horizon):
        return

class NelderMead(otherAlg):
    def Callback(self,X):
        return

    def run(self, x0, horizon):
        self.res = scipy.optimize.minimize(self.func, x0=x0, callback=self.Callback, options={'maxfev': horizon},
                              method='Nelder-Mead')



