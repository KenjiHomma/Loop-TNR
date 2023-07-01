# Loop-TNR
This is a sample python code for Loop-TNR algorithm [S. Yang, Z.-C. Gu, and X.-G. Wen, Phys. Rev. Lett. 118, 110504 (2017)]  for 2D ising model . No guarantee for the validity of the codes.

# Requirements
- Python 3
- numpy
- scipy
- matplotlib
- ncon
  
# How to run
After installing the packages, you can run this code as, 
 ```
python3 Loop-TNR.py 
 ```
By default, The main file "Loop-TNR.py" produces the relative error of free energy density and the central charge of the crtical 2D ising model at bond dimension $\chi =8$. To change some of parameters, such as bond dimensions, number of RG steps, temperature, etc ..., one can add these parameters at the command line.
