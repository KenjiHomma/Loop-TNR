import numpy as np
from ncon import ncon
import scipy.linalg as scl
import scipy.integrate as integrate
from filtering import filtering
from filtering_periodic_mps import filtering_periodic_mps
from loop_optimization import optimize
from plot_spectrum import plot_spectrum
from plot_CFT_data import plot_CFT_data

def Exact_Free_energy(temperature):
        exact_sol = 0
        def funct_to_integrate(theta1,theta2,beta):
            return np.log((np.cosh(2*beta))**2-np.sinh(2*beta)*(np.cos(theta1)+np.cos(theta2)))

        beta = 1/temperature
        integ = integrate.dblquad(funct_to_integrate,0,np.pi,lambda x: 0, lambda x: np.pi,args=([beta]))[0]
        exact_sol = (-1/beta)*((np.log(2)+(1/(2*np.pi**2))*integ))
        return exact_sol
def Ising_tensor(temp):

    beta = 1/temp
    H_local = np.array([[-1,1],[1,-1]])
    M = np.exp(-beta*H_local)
    delta = np.zeros((2,2,2,2))
    delta[0,0,0,0] = 1.
    delta[1,1,1,1] = 1.
    Msr = scl.sqrtm(M)
    T = ncon([delta,Msr,Msr,Msr,Msr],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]])

    Ts = []
    for i in range(4):
        Ts.append(T)
    return Ts


def normalize_T(Ts,g):
    """
    Normalization of 4-index tensors Ts
    """
    for i in range(4):
        Ts[i] = Ts[i]/(g**(1/4))
    return Ts
def LN_renormalization(Ts):
    """
    Renormalizing coarse-grained tensors into new one tensors in square lattice using Lavin-Nave TRG method.
    Arguments:
        Ts: Sets of 3-index tensors 
    Output:
       res_Ts:  Sets of 4-index tensors
    """  
    res_Ts= []

    T1 = ncon([Ts[7],Ts[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[3],Ts[0]],[[-3,1,-2],[-1,1,-4]])
    res_T0 = ncon([T1,T2],[[-1,-2,1,2],[1,2,-3,-4]])

    T1 = ncon([Ts[1],Ts[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[5],Ts[2]],[[-3,1,-2],[-1,1,-4]])
    res_T1 = ncon([T1,T2],[[-1,-2,1,2],[1,2,-3,-4]])

    res_T2 = res_T0.transpose(2,3,0,1)
    res_T3 = res_T1.transpose(2,3,0,1)

    res_Ts.append(res_T0)
    res_Ts.append(res_T1)
    res_Ts.append(res_T2)
    res_Ts.append(res_T3)
    return res_Ts


def LN_TRG_decomp(Ts,chi):
    """
    Decomposing tensors on square lattice into an octagon configuration using Lavin-Nave TRG method.
    
    Arguments:
        Ts: Sets of 4-index tensors
        chi: Bond dimension
    Output:
        LN_decomp: Sets of 3-index tensors 
    """  
    LN_decomp= []
    for i in range(4):

        size1 = np.shape(Ts[i])
        u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)
        if len(s) >  chi :
            u = u[:,:chi]
            s = s[:chi]
            v = v[:chi,:]
        size2 = np.shape(np.diag(s))
        s1 = u@np.sqrt(np.diag(s))
        s2 = np.sqrt(np.diag(s))@v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)]))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]]))

    return LN_decomp
def transfer_matrix(Ts):
    """
    Extracting CFT data from 2 by 2 transfer matrix using Gu-Wen Method.

    Arguments:
        Ts: Sets of 4-index tensors
      
    Output:
        g: normalization factor  
        central_charge: central charge
        scaling_dims: scaling dimensions
    """   
    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])
    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2

    eig , _ =  np.linalg.eig(M)
    eig = -np.sort(-eig)
    g = np.trace(M)

    central_charge = (6/(np.pi))*np.log(eig[0]/(g*g))
    scaling_dims = -(0.5/np.pi)*np.log(eig[1:41]/eig[0])
    return g,central_charge,scaling_dims
def Loop_TNR(Ts, FILT_EPS, FILT_MAX_I, OPT_EPS, OPT_MAX_I, RG_I,chi, temp,solver_eps):
    """
    Loop-TNR main

    """
    G = 1
    Ts = normalize_T(Ts,G)
    exact_f = Exact_Free_energy(temp)
    spectrum_list = []

    c_list = []

    CFT_data_list=[]

    C = 0
    N = 1
    Nplus = 2



    print("\n =============== Loop-TNR starts ===============\n")
    for i in range(RG_I):
        print("\n//----------- Renormalization step:   "+ str(i)+ " -----------\n")

        print("\n * Part 1: Filtering in process\n")
        Ts = filtering(Ts,FILT_EPS, FILT_MAX_I)

        # ----------- Preconditioning for loop-optimization ----------- .  
        # Please comment it out if not necessary.
        eight_tensors = filtering_periodic_mps( LN_TRG_decomp(Ts,chi**2), 1e-12, 100,chi) 
        #----------- Preconditioning ends ----------- 

        eight_tensors_p = LN_TRG_decomp(Ts,chi**2)
        print("\n * Part 2: Optimization in process\n")
        eight_tensors =  optimize(eight_tensors,eight_tensors_p,OPT_MAX_I,OPT_EPS,solver_eps)
        Ts = LN_renormalization(eight_tensors)

        G0 = G
        G,central_c ,scaling_dims= transfer_matrix(Ts)
        Ts = normalize_T(Ts,G)

        C = np.log(G**(1/4))+Nplus*C
        N *= Nplus
        f = -temp*(np.log(G)+2*C)/(2*N)

        print("\n * free energy_error :       " +str(np.abs((exact_f-f)/exact_f))+ "\n")
        print("\n * central charge :       " +str(central_c)+"\n")

         ##### Below,some of physical quantities obtained from Loop-TNR are stored for plot #####
        T = np.reshape(Ts[0],[Ts[0].shape[0]*Ts[0].shape[1],Ts[0].shape[2]*Ts[0].shape[3]])
        u,spectrum,v = np.linalg.svd(T,full_matrices=False)
        count = 101
        spectrum = spectrum[:count]/spectrum[0]

        if len(scaling_dims) is not 40:
            size =40- len(scaling_dims)
            scaling_dims = np.hstack((scaling_dims, np.zeros(size)))

        if len(spectrum) is not 101:
            size = 101- len(spectrum)
            spectrum = np.hstack((spectrum, np.zeros(size)))

        if i > 2:
            c_list.append(central_c)
            CFT_data_list.append(np.real(scaling_dims).tolist())
            spectrum_list.append(list(spectrum))

        
        spectrum_old = spectrum

    CFT_data_list = np.array(CFT_data_list)

    ## Plot the singular value and scaling dimension spectrum
    # Please comment out if not necessary.
    label = ("chi="+str(chi)+"FILT_MAX_I="+str(FILT_MAX_I)+"OPT_MAX_I="+str(OPT_MAX_I)+"solver_eps="+str(solver_eps))
    plot_spectrum(spectrum_list,chi,label)
    plot_CFT_data(c_list, CFT_data_list,chi,label)


    return Ts
import argparse

"""
    A sample implementation of Loop-TNR for 2D square lattice classical Ising model. 
    (S. Yang, Z.-C. Gu, and X.-G.Wen, Loop optimization for
    tensor network renormalization, Phys. Rev. Lett. 118,110504 (2017).)

    This code is used as a comparison target of our paper. Note that no symmetries (C4 rotational or Z2 internal symmetries) are imposed in this implementation)
    Below, we list several parameters for users to tune. 

    Parameters:
    chi :  Bond dimension, generally if one increases \chi, the accuracy of Loop-TNR would be improved.
    temp_ratio : Temperature ratio T/Tc, where Tc refers to the exact transition temperature of 2D classical Ising model.
    RG_step : Number of iterations for RG step.
    FILT_EPS : Truncations threshold for "entanglement filtering". By default,  FILT_EPS is fixed to be 1E-12.
    FILT_MAX_I : Number of iterations for "entanglement filtering". By default,  FILT_MAX_I is fixed to be 100.
    OPT_EPS: Stopping threshold for loop optimization. This threshold refers to the cost function proposed in the original paper. 

    Hyper-parameters:
    OPT_MAX_I: Number of maximum (sweep) iterations for loop optimization. 
    solver_eps: Cut-off ratio for small singular values in the linear-matrix solver. Generally, the smaller the better.

"""
parser = argparse.ArgumentParser(
        description="Simulation of 2D classical Ising model by Loop-TNR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
parser.add_argument("chi", type=int,nargs="?", help="Bond dimension",default=8)
parser.add_argument("temp_ratio", type=float,nargs="?",help="temp ratio",default=1)
parser.add_argument("RG_step", type=int,nargs="?",help="RG_step",default=51)
parser.add_argument("FILT_EPS", type=float,nargs="?",help="FILT_EPS",default=1E-12)
parser.add_argument("FILT_MAX_I", type=int,nargs="?",help="FILT_MAX_I",default=100)
parser.add_argument("OPT_EPS", type=float,nargs="?",help="OPT_EPS ",default= 1E-14)
parser.add_argument("OPT_MAX_I", type=int,nargs="?",help="OPT_MAX_I",default= 30)
parser.add_argument("solver_eps", type=float,nargs="?",help="OPT_MAX_I",default= 1E-12)

args = parser.parse_args()
chi = args.chi
temp_ratio = args.temp_ratio
RG_step = args.RG_step
FILT_EPS =  args.FILT_EPS
FILT_MAX_I = args.FILT_MAX_I
OPT_EPS = args.OPT_EPS
OPT_MAX_I = args.OPT_MAX_I
solver_eps =  args.solver_eps


"""
Setting Local Hamiltonian
Ts: sets of 4-index tensors whose local Hamiltonian is encoded.
"""

temp =  temp_ratio*2/np.log(1+np.sqrt(2))
Ts = Ising_tensor(temp)

_ = Loop_TNR(Ts,FILT_EPS, FILT_MAX_I, OPT_EPS, OPT_MAX_I,RG_step,chi, temp,solver_eps)
