""" Misc. utility functions
"""

import numpy as np
import scipy.optimize as spo

def Nroots(fun, N, xmin=0, dx=0.1):
  # Find the first N roots of fun greater than xmin
  roots = np.empty(N)
  n = 0
  xprev = xmin
  fprev = fun(xmin)
  while n < N:
    xi = xprev + dx
    fi = fun(xi)
    # If sign of function has changed, look for a root in that interval
    if fi * fprev < 0:
      sol = spo.root_scalar(fun, bracket=(xprev, xi))
      roots[n] = sol.root
      n += 1
    # Update previous values
    xprev = xi
    fprev = fi
  return roots
