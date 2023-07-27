import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
import scipy.io as scio

from dolfin import *
from fenics import *
import warnings

# Set communicator (in case running in parallel)
# i am not running in parallel
# comm = MPI.comm_world
# comm_rank = MPI.rank(comm)


# all inline for today bc im fucking cool

# TODO fix your variable names. time step does not represent the time step lol

# define our constants
L = 0.02 # length
t_end = 10.0 # end time
num_pts = 100 # num data points
tol = 1E-6 # tolerance
d_t = t_end/num_pts # time step?
curr_time = 0.0 # what time is it

# actual problem constants
heat_source = Constant(0.0) # following that video im watching so im doing this
T_max = 240 # degrees K?
x_d = 0.005 # start of heat dropoff
#k = 100/L
k = 2500
k_r = 0.15 # from example_DCPD
rho = 980.0 # from example_DCPD
Cp = 1600.0 # from example_DCPD
a = k_r/(rho*Cp) # from example_DCPD

# define our mesh
mesh = IntervalMesh(num_pts, 0, L)

# boundary conditions
left = CompiledSubDomain("near(x[0], side, tol) && on_boundary", side=0.0, tol=tol)
right = CompiledSubDomain("near(x[0], side, tol) && on_boundary", side=L, tol=tol)
bounds = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
bounds.set_all(0)
left.mark(bounds, 1)
right.mark(bounds, 2)
# A_n expression
A_n = Expression("T_max/2 * (1 + tanh(k*(x_d-x[0])))", k=k, degree=1, T_max=T_max, x_d=x_d)

# define our function space
V = FunctionSpace(mesh, "CG", 1)

# define our basis functions
T_test = TestFunction(V) # test function for temp
T_trial = TrialFunction(V) # trial function for temp

ic = Expression("A_n * cos((x[0]*3.14)/L)", degree=1, L=L, A_n=A_n)
# ic = Expression("A_n", degree=1, L=L, A_n=A_n)
ic_old = interpolate(ic, V)

ds = Measure('ds', domain=mesh, subdomain_data=bounds)


# variational problem
problem = T_test*T_trial*dx + a*d_t*dot(grad(T_test), grad(T_trial))*dx \
    - ic_old*T_test*dx

# left and right hand sides
lhs, rhs = lhs(problem), rhs(problem)
# solution storage
solution = Function(V)

# lol jk post processing in real time!
# solving a function now
fig = plt.figure() # reuse the same figure
plt.xlim([0,L])
plt.ylim([0,T_max])
plt.grid()
plt.xlabel('x')
plt.ylabel(r'$T~^oC$')
frames = [] # housing for the frames
for i in range(num_pts):
    curr_time += d_t
    # FEM assmebly
    solve(lhs == rhs, solution)
    ic_old.assign(solution)
    # post processing and animation making
    title = plt.text(0.0085, 250.00, f"Time: {curr_time:1.2f}s")
    img, = plot(solution)
    frames.append([img, title])


ani = animation.ArtistAnimation(fig, frames, interval=50, blit=False)
ani.save("heatAnimation.mp4", writer="ffmpeg")




