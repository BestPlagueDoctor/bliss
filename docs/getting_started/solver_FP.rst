FP Solver
==================
This section provides details of classes and functions of this FP solver.

Main solver
--------------------
.. automodule:: solver_FP
  :members:
  :undoc-members:
  :show-inheritance:

Cure kinetics
------------------------------
The kinetics library defines the cure kinetics model. Some popular models are provided, such as

the first order, 

.. math::
  g(\alpha) = (1 - \alpha),

the N-th order,

.. math::
  g(\alpha)= (1-\alpha)^n,

The Prout Tompkins model,

.. math::
  g(\alpha)=\alpha^m (1-\alpha)^n,

the Prout Tompkins Diffusion model,

.. math::
  g(\alpha)= (1-\alpha)^n \alpha^m \frac{1}{1+\exp \left[C\left(\alpha-\alpha_c\right)\right]}.

A custom cure kinetics model can also be defined by writing your own kinetics class following these provided examples.

.. automodule:: kinetics_library
  :members:
  :undoc-members:
  :show-inheritance:

Post process
------------------------------
.. automodule:: postProcess_FP
  :members:
  :undoc-members:
  :show-inheritance:

