""" Code to simulate a beam according to the Euler-Bernoulli beam theory

The forcing function, beam parameters, resolution, and other settings can be 
    changed in the ##Options section
"""

import os
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import Nroots

## Options
plt_eigfuns = False     # If True, plot eigenfunctions instead of simulating
np.set_printoptions(precision=4)

# Specify the forcing function
input_type = "sweptsin"     # "sin", "sweptsin", or "impulse"
s0 = 1      # q = delta(x-x0)*sin(ws*t)
alpha = -4  # [for input_type sweptsin] d/dt {ws}
ws_t = lambda t: 40 + alpha*t   # [for input_type sweptsin]
ws = 30     # [for input_type sin] frequency of source / forcing function
#x0 = 0.78344      # Location at which forcing function is applied
x0 = 1      # Location at which forcing function is applied

# Parameters of beam
L = 1       # Length of beam
EI = 1      # Stiffness (Young's modulus * 2nd moment of area)
rho = 1     # Linear density

# Resolution & accuracy options
Nn = 25     # Number of terms in series
Nx = 500    # Resolution in x
Nt = 160    # Number of timesteps
t1 = 10     # Final time
# Integration settings
atol = 1e-6 # Integration tolerance
ilimit = 8000   # Max number of intervals for quad

# Plotting options
figsz = (5,3)
spadj = (.09, .11, .98, .91)
to_save = True
fps = Nt / t1
umargin = 0.2   # By what fraction to increase the y-scale beyond the max U
lw = 4          # Beam line width
#tag = "impulse_L"  # Tag for filename
tag = "ss40xL"  # Tag for filename
out_dir = "../tmp/"

## Response u_{x,t} = sum_n{ ubar_{n,t} * X_{n,x} }

# Numerically find eigenvalues
res = lambda lmd: 1 + np.cos(lmd*L)*np.cosh(lmd*L)
lmd_n = Nroots(res, Nn)
# Pre-calculate omega_n
omega_n = lmd_n**2*np.sqrt(EI/rho)
print("Eigenvalues lambda_n:")
print(lmd_n)
print("Natural frequencies omega_n:")
print(omega_n)
# Pre-calculate beta_n
beta_n = (np.cosh(lmd_n*L) + np.cos(lmd_n*L)) / (np.sinh(lmd_n*L)
    + np.sin(lmd_n*L))

# n grid (0-indexed n)
vn = np.arange(Nn)
# Spaces in t & x
vt = np.linspace(0, t1, Nt)
vx = np.linspace(0, L, Nx)

# Calculate eigenfunctions X_n(x),
#   the eigenfunctions evaluated at x_0, X_n(x_0): X_n0,
#   the squared norm of the eigenfunctions, ||X||^2,
#   and the normalized eigenfunctions, X_n(x)/||X||^2: X_nx
X_nx = np.empty((Nn, Nx))
X_n0 = np.empty((Nn, 1))
for n in range(Nn):
    eigfun = lambda x: (np.cosh(lmd_n[n]*x) - np.cos(lmd_n[n]*x)
        - beta_n[n]*(np.sinh(lmd_n[n]*x) - np.sin(lmd_n[n]*x)))
    X_n0[n,0] = eigfun(x0)
    eigfun2 = lambda x: eigfun(x)**2
    N2, *quadout = spi.quad(eigfun2, 0, L, limit=ilimit, epsabs=atol,
        full_output=True)
    if quadout[0] > atol:
        print(78, quadout[0], quadout[-1])
    X_nx[n,:] = eigfun(vx) / N2
print("Done calculating $X_{nx}$")
#print(X_nx.shape)

if plt_eigfuns:
    # Plot function used to find eigenvalues
    vlam = np.linspace(0, 20, 1000)
    fig, ax = plt.subplots(figsize=figsz)
    fig.subplots_adjust(.13, .15, .98, .97)
    ax.plot(vlam, res(vlam), label="res")
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$1 + cos(\lambda L)cosh(\lambda L)$")
    ax.set_ylim(-40, 40)
    ax.grid()
    pname = os.path.join(out_dir, f"eigenvalues.pdf")
    fig.savefig(pname)
    print(f"Eigenvalues plot saved to {pname}")
    # See if the eigenfunctions look right
    fig, ax = plt.subplots(figsize=figsz)
    fig.subplots_adjust(.08, .15, .83, .97)
    ax.plot(vx, X_nx[0, :], label="$X_1$")
    ax.plot(vx, X_nx[1, :], label="$X_2$")
    ax.plot(vx, X_nx[2, :], label="$X_3$")
    ax.plot(vx, X_nx[3, :], label="$X_4$")
    ax.set_xlabel("x")
    ax.grid()
    fig.legend()
    pname = os.path.join(out_dir, f"eigenfunctions.pdf")
    fig.savefig(pname)
    print(f"Eigenfunctions plot saved to {pname}")
    plt.show()
    quit()

# Calculate ubars_{nt}
if input_type == "sin" or input_type == "impulse":
    # For sin & impulse, the convolution was performed analytically
    # Reshape omega so that axis 1 represents n and axis 2 represents t
    wn = omega_n[:,np.newaxis] 
    c0 = s0/(rho*wn) * X_n0
    if input_type == "sin":
        ubar_nt = c0 * (wn*np.sin(vt*ws) - ws*np.sin(vt*wn)) / (wn**2 - ws**2)
    elif input_type == "impulse":
        ubar_nt = c0 * ws*np.sin(vt*wn)
else:
    # Perform convolution with quadrature
    def calc_ubars(n, t): 
        """ Calculate an entry of ubars_{nt}
        ubars = c0 * convolve(f1, f2)
        """
        
        c0 = s0/(rho*omega_n[n]) * X_n0[n,0]
        # Convolution: integrate(f1(t-tau)*f2(tau), (tau, 0, t))
        if input_type == "sweptsin":
            #cnvarg = lambda tau: np.sin(alpha*(t-tau)**2) * np.sin(omega_n[n]*tau)
            cnvarg = lambda tau: np.sin(ws_t(t-tau)*(t-tau)) * np.sin(omega_n[n]*tau)
        # NOTE: This quadrature could possibly be vectorized to speed it up
        #cnvres, *_ = spi.quad(cnvarg, 0, t)
        cnvres, *quadout = spi.quad(cnvarg, 0, t, limit=ilimit, epsabs=atol,
            full_output=True)
        if quadout[0] > atol:
            print(f"At t={t:.2f}, n={n}: quad err = {quadout[0]}")
        return c0 * cnvres, quadout

    # Calculate all of ubars_{nt}
    #   This was set to float32 to save memory, but could likely be reverted
    #   to float64 with no problems
    ubar_nt = np.empty((Nn, Nt), dtype=np.float32)
    for ti in range(Nt):
        for ni in range(Nn):
            ubar_nt[ni, ti], quadout = calc_ubars(vn[ni], vt[ti])
            #print("quad output[0]:")   # For debugging
            #print(quadout[0])
        if (ti+1)%10 == 0:
            print(f"Timestep {ti+1}/{Nt} done")

print("Done calculating $ubar_{nt}$")
#print(60, ubar_nt.shape, X_nx.shape)

# Perform tensor contraction
u_xt = np.einsum("nt,nx->xt", ubar_nt, X_nx)
print("Done calculating $u_{xt}$")

# Set up animated plot
Umax = np.max(np.abs(u_xt))
print(f"\tAll-time max |u|: {Umax}")
U0 = u_xt[:, 0]

fig, ax = plt.subplots(figsize=figsz)
fig.subplots_adjust(*spadj)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_ylim(-(1+umargin)*Umax, (1+umargin)*Umax)
zh, = ax.plot(vx, U0, "k", lw=lw, solid_capstyle="round")
# Input location annotation
ax.annotate("Forcing\nlocation", xy=(x0, 0), xytext=(x0, -Umax*.7),
    arrowprops=dict(facecolor='blue', shrink=0.1), ha="center")

def update_animation(frmi):
    # Update the figure to frame frmi
    U = u_xt[:,frmi]
    ti = vt[frmi]
    frqtag = ""
    if input_type == "sin":
        frqtag = f", frq $\\omega_n={ws}$"
    if input_type == "sweptsin":
        frqtag = f", frq $\\omega_n={ws_t(ti):.2f}$"
    fig.suptitle(f"time $t={ti:.2f}/{t1:.2f}${frqtag}")
    zh.set_ydata(U)
    return zh

anim = FuncAnimation(fig, update_animation, frames=Nt,
    interval=1000/fps, repeat_delay=500)
pname_anm = os.path.join(out_dir, f"beam_{tag}.gif")

if to_save:
    anim.save(pname_anm, writer='imagemagick', fps=fps)
    print(f"Animation saved to {pname_anm}")

plt.show()

