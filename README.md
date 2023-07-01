# Loop-TNR
This is a sample python code for Loop-TNR algorithm [S. Yang, Z.-C. Gu, and X.-G. Wen, Phys. Rev. Lett. 118, 110504 (2017)]  for 2D ising model . No guarantee for the validity of the codes.

# Requirement 
Python 3
numpy
scipy
matplolib
ncon

# How to run
After installing the packages, you can run this code as, 
 ```
python3 Loop-TNR.py 
 ```
By default, The main file "Loop-TNR.py" produces the relative error of free energy density and the central charge of the crtical 2D ising model at bond dimension $\chi =8$. To change some of parameters, such as bond dimensions, number of RG steps, temperature, etc ..., one can add these parameters at the command line.

positional arguments:
  chi         Bond dimension (default: 8)
  temp_ratio  temp ratio (default: 1)
  RG_step     RG_step (default: 51)
  FILT_EPS    FILT_EPS (default: 1e-12)
  FILT_MAX_I  FILT_MAX_I (default: 100)
  OPT_EPS     OPT_EPS (default: 1e-14)
  OPT_MAX_I   OPT_MAX_I (default: 30)
  solver_eps  OPT_MAX_I (default: 1e-12)
