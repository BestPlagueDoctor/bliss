# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# Contains a series of useful functions for post-processing data

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy import interpolate as itp
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.animation as animation
import scipy.io as scio

class Post_process():
    """
    A post-process class for creating report, as well as plotting the evolution of alpha and T at given points, 
    computing frontal speed, and generating the animation of evolution of alpha and T.

    """

    def __init__(self, results_data, fpSolver=None,pts=[],pltTpeak=False, save_dir="out", save_suffix="example",anim_frames=100, fid=0):
        """
        Initializes the Post process using the supplied parameters

        Parameter
        ---------
        results_data - dictionary
           results returned by FP_solver.solve(), including  x_data, t_data, T_data, alpha_data, T_peak

        fpSolver - object
            The FP solver, needed for generating the report

        pts - list
            list of x-coordinates (float, unit (m)), extract the T/alpha vs time at pts. Default is empty.  

        pltTpeak - bool
            plot the maximum temperature along x-direction, default is False
        
        save_dir - str
            The relative path to which front position vs time plot is saved. Default is "out"

        save_suffix - str
            Used to name the save file for front position vs time, which takes the form: "frontPos_{save_suffix}.png"
            Default is "example"

        anim_frame - int
            Number of frames in animation. Default is 100
        
        fid - int
            The id for plot
        
        """

            # Set thermo-chemical properties
        self.x_data = results_data['x_data']
        self.t_data = results_data['t_data']
        self.T_data = results_data['T_data']
        self.alpha_data = results_data['alpha_data']
        self.T_peak=results_data['T_peak']
        self.save_dir = save_dir
        self.save_suffix = save_suffix
        self.fid = fid
        self.anim_frames = anim_frames
        self.fpSolver = fpSolver
        self.pts=pts
        self.pltTpeak=pltTpeak
        self.figID=0
    def computeFrontSpeed(self):
        """
        Computes the front speed by tracking the location where alpha = 0.5 during the simulation.
        Returns the front speed.

        Returns
        -------
        Vfem - float
            The front speed computed from the slope of a linear fit through the front position vs time curve

        rptStr - string    
            A string in html format for generating the report.

        """
        #     :return: The front speed computed from the slope of a linear fit through the front position vs time curve
        # :rtype: float
        # Make save directory
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Get a list of the front locations by location of alpha = 0.5
        frontLoc_arr = []
        for i, t in enumerate(self.t_data[:, 0]):
            frontLoc = itp.interp1d(
                self.alpha_data[i, :], self.x_data[i, :], kind='nearest', assume_sorted=False, bounds_error=False)
            frontLoc_arr.append(float(frontLoc(0.5)))

        # Remove data points where nan values are
        data_arr = np.concatenate(
            (self.t_data[:, 0].reshape(-1, 1), np.array(frontLoc_arr).reshape(-1, 1)), axis=1)
        data_arr = data_arr[~np.isnan(data_arr).any(axis=1), :]

        # Compute front velocity as linear fit through data if possible, else no front found
        if np.any(data_arr):
            V_fem, bbb = np.polyfit(data_arr[:, 0], data_arr[:, 1], 1)
        else:
            V_fem = 0
            frontLoc_arr = np.zeros_like(self.t_data)

        # Plot result and print front velocity
        self.fid+=1
        fig = plt.figure(self.fid)
        ax = fig.add_subplot()
        ax.plot(self.t_data[:, 0], frontLoc_arr, '--', linewidth=2, label='fem')
        ax.set_xlabel('t (s)', fontsize=13)
        ax.set_ylabel(r'$X_{fp}$ (m)', fontsize=13)
        ax.legend()


        save_path = os.path.join(
            self.save_dir, "frontPos" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
        fname=os.path.join( "frontPos" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
        plt.savefig(save_path, bbox_inches='tight')

        rptStr=('      <p>The frontal position \( x_{fp}(t)\) is extracted as the location where \( \\alpha = 0.5 \): </p>\n')
        rptStr+=('      <img src="%s" alt="frontal position" width="400" height="300">\n'%(fname))
        self.figID+=1
        rptStr+=('      <figcaption><span class="fig-label">Fig. %d:</span> The frontal position vs time.</figcaption>'%(self.figID))
        rptStr+=('      <p>The front velocity is obtained by linear regression of the front position vs time curve, and the value is \( V_{fp} = %.4f \) (mm/s).</p>\n'%(V_fem*1000))

        print("FP velocity %.3e (mm/s)" % (V_fem*1000))

        return V_fem, rptStr


    def frontAnimation(self):
        """
        Save an animation of the temperature and degree of cure fronts

        """

        # Make save directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        numFrame = self.t_data.shape[0]  # number of frames

        # Create animation and save
        self.fid+=1
        fig = plt.figure(self.fid)
        ax = fig.add_subplot()
        ax.vlines(0.005,0,1)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel(r'$\alpha$', fontsize=13)
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$T~^oC$', fontsize=13)
        ims = []
        frames = np.arange(0, numFrame, int(np.ceil((numFrame-1)/self.anim_frames)))
        frames[-1] = numFrame-1
        for i in frames:
            im1, = ax2.plot(self.x_data[i, :], self.T_data[i, :] - 273, '--r', linewidth=2)
            im2, = ax.plot(self.x_data[i, :], self.alpha_data[i, :], '--b', linewidth=2)
            title = ax.text(0.5, 1.05, "time = {:.2f}s".format(self.t_data[i, 0]),
                            size=plt.rcParams["axes.titlesize"],
                            ha="center", transform=ax.transAxes, )
            ims.append([im1, im2, title])  # im1,
        ax.legend([im1, im2], [r'$T$', r'$\alpha$'])  # im1,
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=False)

        save_path = os.path.join(self.save_dir, "frontAnim" +
                                ("_"+self.save_suffix if self.save_suffix else "") + ".mp4")
        ani.save(save_path, writer='ffmpeg')

    def evoluAtPts(self):
        """
        Plot the evolution of alpha and T at given points.

        :return: A string in html format for generating the report.
        :rtype: string

        """
        if not self.pts:
            print("pts list is empty!")
            return '\n'
        else:
            pts=np.array(self.pts)
            alpha_elv=np.zeros((pts.shape[0], self.t_data.shape[0]))
            T_elv=np.zeros((pts.shape[0], self.t_data.shape[0]))
            for i, t in enumerate( self.t_data[:, 0]):
                alpha_fun = itp.interp1d(
                    self.x_data[i, :],  self.alpha_data[i, :], kind='linear', assume_sorted=False, 
                    bounds_error=False, fill_value="extrapolate")
                T_fun = itp.interp1d(
                    self.x_data[i, :],  self.T_data[i, :], kind='linear', assume_sorted=False, 
                    bounds_error=False, fill_value="extrapolate")
                alpha_elv[:,i]=alpha_fun(pts)
                T_elv[:,i]=T_fun(pts)

            self.fid+=1
            fig = plt.figure(num=self.fid,figsize=(12.8,4.8))
            ax1= fig.add_subplot(1,2,1)
            ax2=fig.add_subplot(1,2,2)
            ax1.set_xlabel('t (s)', fontsize=13)
            ax1.set_ylabel(r'$\alpha$', fontsize=13)
            ax1.xaxis.set_tick_params(labelsize=13)
            ax2.set_xlabel('t (s)', fontsize=13)
            ax2.set_ylabel(r'$T~^oC$', fontsize=13)
            ax2.xaxis.set_tick_params(labelsize=13)
            for  i, pt in enumerate(pts):
                ax1.plot( self.t_data[:, 0],alpha_elv[i,:],'--', linewidth=2,label=r'$x=%.2f$ (mm)'%(pt*1000))
                ax2.plot( self.t_data[:, 0],T_elv[i,:],'--', linewidth=2,label=r'$x=%.2f$ (mm)'%(pt*1000))
            ax1.legend( fontsize=13)
            ax2.legend( fontsize=13)

            save_path = os.path.join(
                self.save_dir, "evolu" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
            fname=os.path.join( "evolu" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
            plt.savefig(save_path, bbox_inches='tight')
            strpts=', '.join([ '\( x = %.2f \) (mm)'%(e*1000) for e in self.pts])
            rptStr='      <p>The evolution of degree of cure and temperature at the points %s are shown below,</p>\n'%(strpts)
            rptStr+=('      <img src="%s" alt="evolution" width="600" height="225">\n'%(fname))
            self.figID+=1
            rptStr+=('      <figcaption><span class="fig-label">Fig. %d:</span> Evolution of degree of cure and temperature at specified points.</figcaption>'%(self.figID))
            return rptStr

    def plotT_peak(self):
        """
        Plot the evolution of alpha and T at given points.

        :return: A string in html format for generating the report.
        :rtype: string

        """
        if not self.pltTpeak:
            return '\n'
        else:
            self.fid+=1
            fig = plt.figure(self.fid)
            ax = fig.add_subplot()
            ax.plot(self.x_data[0, :], self.T_peak-273, '--', linewidth=2)
            ax.set_xlabel('x (m)', fontsize=13)
            ax.set_ylabel(r'$T_{peak}~(^oC)$', fontsize=13)
            ax.legend()

            save_path = os.path.join(
                self.save_dir, "Tpeak" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
            fname=os.path.join( "Tpeak" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
            plt.savefig(save_path, bbox_inches='tight')
            rptStr='      <p>Spatial variation of the thermal spike are shown below,</p>\n'
            rptStr+=('      <img src="%s" alt="Tpeak" width="400" height="300">\n'%(fname))
            self.figID+=1
            rptStr+=('      <figcaption><span class="fig-label">Fig. %d:</span> The maximum temperature along x-direction.</figcaption>'%(self.figID))
            return rptStr




    def generateReport(self):
        """ Create the report file in html format.

        1. Save an animation of evolution of the temperature and degree of cure

        2. Computes the front speed by tracking the location where alpha = 0.5 during the simulation and save the front position vs time plot

        3. Plot of the evolution of degree of cure and tempetrue at specified points (pts).
        """    
        if not self.fpSolver:
            print("Error: The FP solver object is needed for generating a report!")
            return
        save_path_html = os.path.join(self.save_dir, "Report" +
                        ("_"+self.save_suffix if self.save_suffix else "") + ".html")
        save_path_ani = os.path.join("frontAnim" +
                                ("_"+self.save_suffix if self.save_suffix else "") + ".mp4")
        save_path_xfp = os.path.join("frontPos" + ("_"+self.save_suffix if self.save_suffix else "") + ".png")
        self.frontAnimation()
        V_fem, VfStr=self.computeFrontSpeed() # 
        ptsStr=self.evoluAtPts()
        TpeakStr=self.plotT_peak()

        with open(save_path_html, 'w') as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('  <head>\n')
            f.write('    <title>FP simulation report</title>\n')
            f.write('    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n')
            f.write('    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"></script>\n')
            f.write('  </head>\n')
            f.write("  <body>\n")
            f.write("    <section>\n")
            f.write("      <h1>FP simulation report</h1>\n")
            f.write("      <p> This is the report of the FP problem you just solved.</p>\n")
            f.write("    </section>\n")
            f.write("    <section>\n")
            f.write("      <h2>Problem description</h2>\n")
            f.write("      <p>The propagation of a polymerization front can be described by a thermo-chemical model based on two coupled reaction-diffusion equations expressed in terms of the temperature \( T \) and degree of cure \( \\alpha \) as follows,</p>\n")
            f.write("      <div id=\"eq:ge\">\n")
            f.write("        $$\n")
            f.write("        \\begin{equation}\\label{eq:ge}\n")
            f.write("          \\left\\{\\begin{array}{l}\n")
            f.write("            \\kappa \\nabla^2 T+\\rho H_r \\frac{\\partial \\alpha}{\\partial t}=\\rho C_p \\frac{\\partial T}{\\partial t}, \\\\\n")
            f.write("            \\frac{\\partial \\alpha}{\\partial t}= A \\exp \\left(-\\frac{E}{R T}\\right) g(\\alpha), \n")
            f.write("          \\end{array}\\right.\n")
            f.write("        \\tag{1}\n")
            f.write("        \\end{equation}$$\n")
            f.write("      </div>\n")
            f.write("      <p> where the first equation describes heat diffusion, and the second equation describes the cure kinetics of the FP reaction.</p>\n")
            f.write("      <p>\\( g(\\alpha) \\) is the reaction model, which is chosen from the <em>kinetics_library</em> module as: </p>\n")
            f.write("      <div id=\"eq:KM\">\n")
            f.write("        $$\n")
            f.write("        \\begin{equation}\n")
            f.write(self.fpSolver.kinetics_model.KMexpression)
            f.write("        \\tag{2}\n")
            f.write("        \\end{equation}\n")
            f.write("        $$\n")
            f.write("      </div>\n")
            f.write('      <p>The descriptions and values of the variables in <a href="#eq:ge">Eq. (1)</a> and <a href="#eq:KM">Eq. (2)</a> are shown below based on your input:</p>\n')
            f.write('      <table id="tab:param">\n')
            f.write('        <thead>\n')
            f.write('          <tr>\n')
            f.write('            <th>Variable</th>\n')
            f.write('            <th>Description</th>\n')
            f.write('            <th>Value</th>\n')
            f.write('          </tr>\n')
            f.write('        </thead>\n')
            f.write('        <tbody>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( \\kappa \\) \\( \\left(\\frac{\\mathrm{w}}{\\mathrm{mK}}\\right)\\)</td>\n')
            f.write('            <td>Thermal conductivity </td>\n')
            f.write('            <td>%.3f</td>\n'%(self.fpSolver.k))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( \\rho \\) \\( \\left(\\frac{\\mathrm{kg}}{\\mathrm{m}^3}\\right) \\)</td>\n')
            f.write('            <td>Density </td>\n')
            f.write('            <td>%.1f</td>\n'%(self.fpSolver.rho))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( C_p \\) \\( \\left(\\frac{\\mathrm{J}}{\\mathrm{kg}\\cdot K}\\right) \\)</td>\n')
            f.write('            <td>Specific heat </td>\n')
            f.write('            <td>%.1f</td>\n'%(self.fpSolver.Cp))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( H_r \\) \\( \\left(\\frac{\\mathrm{J}}{\\mathrm{kg}}\\right) \\)</td>\n')
            f.write('            <td>Total enthalpy of reaction </td>\n')
            f.write('            <td>%.2e</td>\n'%(self.fpSolver.H))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>A \\( \\left(s^{-1}\\right) \\)</td>\n')
            f.write('            <td>Pre-exponential factor </td>\n')
            f.write('            <td>%.2e</td>\n'%(self.fpSolver.A))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>E \\( \\left(\\frac{\\mathrm{J}}{\\mathrm{mol}}\\right) \\)</td>\n')
            f.write('            <td>Activation energy </td>\n')
            f.write('            <td>%.4e</td>\n'%(self.fpSolver.E))
            f.write('          </tr>\n')
            f.write(self.fpSolver.kinetics_model.para)
            f.write('        </tbody>\n')
            f.write('      </table>\n')
            f.write('      <p>The 1D FEM simulation settings are shown below:</p>\n')
            f.write('      <table id="tab:simSet">\n')
            f.write('        <thead>\n')
            f.write('          <tr>\n')
            f.write('            <th>Variable</th>\n')
            f.write('            <th>Description</th>\n')
            f.write('            <th>Value</th>\n')
            f.write('          </tr>\n')
            f.write('        </thead>\n')
            f.write('        <tbody>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( T_0 \\) \\( \\left(^oC \\right) \\)</td>\n')
            f.write('            <td>Initial temperature </td>\n')
            f.write('            <td>%.2f</td>\n'%(self.fpSolver.T0-273.15))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( \\alpha_0 \\)</td>\n')
            f.write('            <td>Initial degree of cure</td>\n')
            f.write('            <td>%.3f</td>\n'%(self.fpSolver.alpha0))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( T_{trig} \\) \\( \\left(^oC \\right) \\)</td>\n')
            f.write('            <td>Trigger temperature </td>\n')
            f.write('            <td>%.1f</td>\n'%(self.fpSolver.T_trig-273.15))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( t_{trig} \\)</td>\n')
            f.write('            <td>Duration of trigger (s)</td>\n')
            f.write('            <td>%.2f</td>\n'%(self.fpSolver.t_trig))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( x_{max} \\) (m)</td>\n')
            f.write('            <td>Length of domain   </td>\n')
            f.write('            <td>%.2e</td>\n'%(self.fpSolver.L))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( t_{end} \\)  (s)</td>\n')
            f.write('            <td>Time span of the simulation</td>\n')
            f.write('            <td>%.1f</td>\n'%(self.fpSolver.t_end))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( \\Delta t (s)\\)</td>\n')
            f.write('            <td> Time step </td>\n')
            f.write('            <td>%.2e</td>\n'%(self.fpSolver.delta_t))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>\\( \\Delta x \\) (m)</td>\n')
            f.write('            <td>Element size </td>\n')
            f.write('            <td>%.2e</td>\n'%(self.fpSolver.h_x))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>output_t</td>\n')
            f.write('            <td>Save frequency in time</td>\n')
            f.write('            <td>%d</td>\n'%(self.fpSolver.outputFreq_t))
            f.write('          </tr>\n')
            f.write('          <tr>\n')
            f.write('            <td>output_x</td>\n')
            f.write('            <td>Save frequency in space</td>\n')
            f.write('            <td>%d</td>\n'%(self.fpSolver.outputFreq_x))
            f.write('          </tr>\n')
            f.write('        </tbody>\n')
            f.write('      </table>\n')
            f.write('    </section>\n')
            f.write('    <section>\n')
            f.write('      <h3>Simulation results</h3>\n')
            f.write('      <p>The evolution of temperature \(T \) and degree of cure \( \\alpha \): </p>\n')
            f.write('      <video controls width="400" height="300">\n')
            f.write('        <source src="%s" type="video/mp4">\n'%(save_path_ani))
            f.write('      </video>\n')
            f.write(VfStr)
            f.write(ptsStr)
            f.write(TpeakStr)
            f.write('    </section>\n')
            f.write('  </body>\n')
            f.write('<article>\n')

            f.write('\t<h4>Geubelle Research Group</h4>\n')
            f.write('\t<ul>\n')
            f.write('\t\t<li>Qibang Liu - <a href="mailto:qibang@illinois.edu">qibang@illinois.edu</a></li>\n')
            f.write('\t\t<li>Michael Zakoworotny - <a href="mailto:mjz7@illinois.edu">mjz7@illinois.edu</a></li>\n')
            f.write('\t\t<li>Philippe Geubelle - <a href="mailto:geubelle@illinois.edu">geubelle@illinois.edu</a></li>\n')
            f.write('\t\t<li>Aditya Kumar - <a href="mailto:aditya.kumar@ce.gatech.edu">aditya.kumar@ce.gatech.edu</a></li>\n')
            f.write('\t</ul>\n')
            f.write('</article>\n')

            f.write('<footer>\n')
            f.write('\t<small>&copy; 2023 UIUC. All Rights Reserved.</small>\n')
            f.write('</footer>\n')

            f.write('</html>\n')


def saveResults(save_data, save_dir, mat_name):
    """ Saves the supplied dictionary save_data to a matlab data file

    :param save_data: Results data to be saved 
    :type save_data: dictionary

    :param save_dir: path to save results
    :type save_dir: string

    :param mat_name: file name to save results, do not include .mat extension
    :type mat_name: string
    """

    results_file = os.path.join(save_dir, mat_name+".mat")
    scio.savemat(results_file, save_data)


def loadResults(load_dir, mat_name):
    """ Unpacks the supplied .mat file into a dictionary and returns

    :return: dictionary with variable names as 'x_data', 't_data', 'alpha_data', 'T_data', 'T_peak', are loaded as
    :rtype: dictionary 
    """

    results_file = os.path.join(load_dir, mat_name+".mat")
    load_data = scio.loadmat(results_file)
    return load_data
