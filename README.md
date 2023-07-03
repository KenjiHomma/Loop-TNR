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
By default, The main file "Loop-TNR.py" produces the relative error of free energy density, conformal data and singular value spectrums of the crtical 2D ising model with Bond dimension $\chi =8$. The computation might take some times depending on the environment. (It ends in less than a minute for MacBook Air M1, 2020.) Once computation is done, RG step dependences of the conformal data and singualr value spectrums will be plotted and saved in /CFTdata001 and /spectrum001, respectively.

To change some of parameters, such as bond dimensions, number of RG steps, temperature, etc ..., one can add these parameters at the command line. For example, to increase the bond dimension to $\chi=16$
 ```
python3 Loop-TNR.py 16
 ```

Although this sample code is already practical for studying the 2d classical statistical model, we have proposed alternative optimization method in arXiv:2306.17479. If you are interested in, please read our paper.

