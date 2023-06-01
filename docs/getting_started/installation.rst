Installation
==========================

This solver is developed based on FENICS, which is run on a Linux-based platform. It can, however, also be run on Windows systems.

Installing WSL for Windows users
---------------------------------
This section is for Windows users only. For Linux or Mac OS users, please skip this section.

For Windows users, `WSL`_ (Windows Subsystem for Linux) will be needed. 

To install WSL, open PowerShell or Windows Command Prompt in administrator mode by searching in the Start menu, right-clicking it, and selecting "Run as administrator." Enter the command: 

.. code:: PowerShell

  wsl --install

and then restart your machine. For more details on WSL's installation, please refer to https://learn.microsoft.com/en-us/windows/wsl/install.

After installing WSL, open a Linux terminal shell by going to the Start menu and running 'Ubuntu', from which a Unix account name and password should be set first (The password can be empty).

Then please follow the below sections to install FENICS.

.. _`WSL` : https://learn.microsoft.com/en-us/windows/wsl/install

Install conda
---------------------------------
We recommend using conda to install fenics.

Check whether conda is already installed on your machine. Open a terminal shell and input:

.. code:: bash

  which conda

This command will display the path where conda is installed. If conda is not found, install it using Mambaforge3:

- Linux Users:

.. code:: bash

  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
  bash Mambaforge-Linux-x86_64.sh -b -p ~/mambaforge3

- Macintosh Users with Intel processors:

.. code:: bash

  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh
  bash Mambaforge-MacOSX-x86_64.sh -b -p ~/mambaforge3

- Macintosh Users with Apple Silicon processors:

.. code:: bash

  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh
  bash Mambaforge-MacOSX-arm64.sh -b -p ~/mambaforge3

With Mambaforge installed in your home directory, export PATH so that it may be used:

.. code:: bash

  export PATH=$HOME/mambaforge3/bin:$PATH

Now that we can execute mamba, initialize it and then exit the terminal:

.. code:: bash

  mamba init
  exit

and then restart a terminal and check if conda is installed to use or not:

.. code:: bash

  which conda

Install Fenics
---------------------------------

Create a virtual environment for FENICS and activate it:

.. code:: bash

  conda create --name fenics -q -y
  conda activate fenics

List all of the versions of fenics available on your platform with conda:

.. code:: bash

  conda search fenics --channel conda-forge

Install FENICS. In 2018, work started on FEniCSx: the new version of the FEniCS library. However, our FP solver is developed in the old version of FEniCS, whose installation uses the following command:

.. code:: bash

 conda install -c conda-forge fenics mshr=2019.1.0=py38h255090a_2 fenics-dijitso fenics-dolfin fenics-ffc fenics-fiat fenics-libdolfin fenics-ufl

Install other dependencies
---------------------------------

Activate the fenics virtual environment:

.. code:: bash

  conda activate fenics

Check whether ffmpeg is installed (used for post-processing):

.. code:: bash

  which ffmpeg

If ffmpeg is not installed, install it:

.. code:: bash

  conda install ffmpeg


Install the other python modules:

.. code:: bash

  pip install matplotlib scipy IPython

Using Fenics
---------------------------------

Each time we use fenics, activation of the fenics virtual environment is needed:

.. code:: bash
  
  conda activate fenics


Uninstall the fenics environment
---------------------------------

If you do not use fenics anymore, you may uninstall it:

.. code:: bash
  
  conda deactivate   # if 'fenics' was currently activated
  conda remove --name fenics --all
