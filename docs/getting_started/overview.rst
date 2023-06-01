Overview
================

The propagation of a polymerization front can be described by a thermo-chemical model based on two coupled reaction-diffusion partial differential equations (PDEs) expressed in terms of the temperature :math:`T` and degree of cure :math:`\alpha` as follows,

.. math::

    \begin{equation}
    \left\{\begin{array}{l}
    \kappa \nabla^2 T+\rho H_r \frac{\partial \alpha}{\partial t}=\rho C_p \frac{\partial T}{\partial t}, \\
    \frac{\partial \alpha}{\partial t}= A \exp \left(-\frac{E}{R T}\right) g(\alpha),
    \end{array}\right.
    \end{equation}

where the first equation describes heat diffusion, and the second equation describes the cure kinetics of the FP reaction.

This Frontal Polymerization Solver (FPS, 1D) is developed based on `FENICS`_, a popular open-source computing platform for solving PDEs.

.. _`FENICS` : https://fenicsproject.org/

