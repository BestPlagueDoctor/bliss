Usage of FPS
================

How to run FPS?
------------------

First, activate the fenics virtual environment:

.. code:: bash

  conda activate fenics

Change to the path where the code files (example_DCPD.py, etc.) are placed

.. code:: bash

  cd /desired/path # for Linux/Mac
  cd /mnt/c/foldername  # for WSL, here c is the disk name, like C:\

Note: You should ensure the python file to be run is in the same folder as the files solver_FP.py, kinetics_library.py, and postProcess_FP.py.

Run the script 'test_FP.py'_ from the command line with:

.. code:: bash

    python test_FP.py  #Rrun the DCPD example 

This will run a unit test to confirm that you have successfully installed fenics and the code is working properly on your machine. If successful, the following output should appear at the end of the command line ::

    FP velocity 1.515e+00 (mm/s)
    .
    ----------------------------------------------------------------------
    Ran 1 test in 19.146s
    
    OK


We have included two example scripts that demonstrate the capabilities of this code. Shown below are implementations for simulating frontal polymerization of DCPD and COD, in the scripts `example_DCPD.py`_ and `example_COD.py`_, respectively. Now run the python script for FP of DCPD:

.. code:: bash

    python example_DCPD.py  # run the DCPD example 

After the simulation is completed, one simulation report file  './out_DCPD/Report_DCPD.html' (`example report  <../_static/Report_DCPD.html>`_) is automatically generated and can be opened in a browser.

.. _example_DCPD.py:

example_DCPD.py 
------------------
.. literalinclude:: ../../FPS/example_DCPD.py
   :language: python

.. _example_COD.py:

example_COD.py 
------------------
.. literalinclude:: ../../FPS/example_COD.py
   :language: python

.. Report example 
.. ------------------

.. .. include:: ./Report_DCPD.html
