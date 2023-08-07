# %%
# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# An example of running the FP solver for COD

# %%
import os
from solver_FP import FP_solver
from kinetics_library import Prout_Tompkins
from postProcess_FP import Post_process, saveResults, loadResults


# Switch to current directory
try:
    os.chdir(os.path.dirname(__file__))
except:
    print("Cannot switch directories")

# Thermo-chemical properties
kr = 0.133     # thermal conductivity, W/m-K
rhor = 882.0    # density, kg/m^3
Cpr = 1838.0    # specific heat, J/kg-K

# Nature values
Hr = 219000    # enthalpy, J/kg
A_ = 2.13e19    # pre-exponential constant, 1/s
Er = 132000.0   # activation energy J/mol
n = 2.517       # cure kinetics parameter
m = 0.817       # cure kinetics parameter

# Create an object for the cure kinetics model
pt_model = Prout_Tompkins(n, m)

# Initial conditions for temperature and degree of cure
T0 = 10         # deg C
alpha0 = 0.02

# Length of domain
L = 0.005

# Thermal trigger
T_trig = 130  # deg C
t_trig = 2.5  # s

# Initialize the solver with the parameters
solver = FP_solver(kr, rhor, Cpr, Hr, A_, Er, pt_model, T0, alpha0,
                   T_trig=T_trig, t_trig=t_trig, L=L, dt=0.25e-3, h_x=0.5e-5)

# Run the solver until 10 s
t_end = 20
results_data = solver.solve(
    t_end, outputFreq_t=40, outputFreq_x=1)

# Post-process - generate the report, compute the front speed, and save an animation of the temperature and alpha fronts
save_dir = "out_COD"  # path to save results.
pltTpeak_ = True  # plot the maximum temperature along x-direction
# create post process object
post_pro = Post_process(results_data, solver, pts=[], pltTpeak=pltTpeak_,
                        save_dir=save_dir, save_suffix="COD", anim_frames=100, fid=0)
post_pro.generateReport()  # generate report

# %%
# Pack the results for future use
saveResults(results_data, save_dir, "data_COD")  # save to .mat file
loadedResults = loadResults(save_dir, "data_COD")  # load from .mat file
