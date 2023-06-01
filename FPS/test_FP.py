# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# Unittest of the FP solvers
# Run this script to ensure that everything is working properly

import unittest
from solver_FP import FP_solver
from kinetics_library import Prout_Tompkins_Diffusion
from postProcess_FP import Post_process


class TestFPSolver(unittest.TestCase):

    def test_DCPDFrontVel(self):
        """ Unittest of the front velocity of DCPD
        """

        # Thermo-chemical properties
        kr = 0.15      # thermal conductivity, W/m-K
        rhor = 980.0    # density, kg/m^3
        Cpr = 1600.0    # specific heat, J/kg-K

        # Nature values
        Hr = 350000    # enthalpy, J/kg
        A_ = 8.55e15    # pre-exponential constant, 1/s
        Er = 110750.0   # activation energy J/mol
        n = 1.72        # cure kinetics parameter
        m = 0.77        # cure kinetics parameter
        Ca = 14.48      # cure kinetics parameter
        alpha_c = 0.41  # cure kinetics parameter

        # Create an object for the cure kinetics model
        ptd_model = Prout_Tompkins_Diffusion(n, m, Ca, alpha_c)

        # Initial conditions for temperature and degree of cure
        T0 = 20         # deg C
        alpha0 = 0.01

        # Length of domain
        L = 0.01

        # Initialize the solver with the parameters
        solver = FP_solver(kr, rhor, Cpr, Hr, A_, Er,
                           ptd_model, T0, alpha0, L=L, h_x=2e-5)

        # Run the solver until 10 s
        t_end = 10
        results_data = solver.solve(
            t_end, outputFreq_t=10, outputFreq_x=1)

        # Post-process - compute the front speed, and save an animation of the temperature and alpha fronts
        post_pro = Post_process(results_data, solver, save_dir="")
        V_front,rptStr = post_pro.computeFrontSpeed()

        V_analytical = 0.001517  # m/s, from analytical expression in Kumar, 2021
        # Check if front speed is equal to within 0.01 mm/s
        self.assertAlmostEqual(V_front, V_analytical, 5)


if __name__ == "__main__":
    unittest.main()
