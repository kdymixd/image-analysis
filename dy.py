"""Physical constants of Dysprosium 162"""

import scipy.constants as ct
from numpy import pi


hbar=ct.hbar
c=ct.c
lambdaDyRed=626.082*1e-9 #Red transition of Dysprosium
lambdaDyBlue=421.290*1e-9 #Blue transition of Dysprosium
delta = 0
epsilon = 7/15

sigma0 = 3*lambdaDyBlue**2/(2*pi)*epsilon 


if __name__ == "__main__":
    print(sigma0)