# beam_EB
Simulation of a cantilever beam according to Euler-Bernoulli beam theory with time-varying applied point loads. 

Figures and animations are found in the /figs/ and /animations/ directories, respectively.

## Background
Euler-Bernoulli beam theory is a simplified model for flexural motion of thin
beams. The Euler-Bernoulli beam equation is as follows:
$$\frac{\partial^2}{\partial x^2}(EI\frac{\partial^2 u}{\partial x^2}) = -\rho\frac{\partial^2 u}{\partial t^2} + q$$

The details of solution and implementation with this code are presented in [1].

At the moment, this code only supports clamped-free boundary conditions. Changing the boundary conditions would require re-deriving the eigenfunctions, which would change the generallized finite integral transform.

## Setup
beam_EB was written in Python 3.10. Backwards-compatability with older 
versions of Python has not been verified.

To install required packages, run
```
$ python -m pip install -r requirements.txt
```

## References
[1] J. K. Black and J. Blackham, "Solution of Euler-Bernoulli Beam Equation by Integral Transform Method," Journal of Applied Engineering Mathematics, Volume 10, December 2023. [Online]. Available: https://www.et.byu.edu/~vps/ME505/AAEM/V10-08.pdf