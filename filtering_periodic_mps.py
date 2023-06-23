import numpy as np
from ncon import ncon

def pbc_moving_left(s,N):
    s = s-1
    if(s== -1):
        s = N-1
    return s
def pbc_moving_right( s, N ):
    s = s+1
    if s == N:
        s=0
    return s

def filtering_right_loop(Ts,eps, MAX_I, s ):

    SIZE = np.shape(Ts[s])
    L = np.eye(SIZE[0])
    U, sv, V = np.linalg.svd(L,full_matrices=False)
    sv = sv/ np.linalg.norm(sv)

    var = 1
    I = 0
    while eps<var and I<MAX_I:
        L = filtering_right_single_loop(Ts, L, s)
        U, sp, V = np.linalg.svd(L,full_matrices=False)
        sp = sp/np.linalg.norm(sp)
        var =  np.linalg.norm(sv-sp)
        sv = sp
        I = I +1
    return L

def filtering_right_single_loop(Ts,L,s):
    size = len(Ts)
    for i in range(size):

        T = ncon([L,Ts[s]], [[-1,1],[1,-2,-3]])
        _, L = np.linalg.qr(T.reshape([T.shape[0]*T.shape[1],T.shape[2]], order='F'))
        s = pbc_moving_right(s,size)

    L = L/np.max(abs(L))
    return L

def filtering_left_loop(Ts,eps, MAX_I, s ):
    SIZE = np.shape(Ts[s])
    R = np.eye(SIZE[2])
    U, sv, V = np.linalg.svd(R,full_matrices=False)
    sv = sv/ np.linalg.norm(sv)
    var = 1
    I = 0
    while eps<var and I<MAX_I:
        R = filtering_left_single_loop(Ts, R, s)
        U, sp, V = np.linalg.svd(R,full_matrices=False)
        sp = sp/np.linalg.norm(sp)
        var =  np.linalg.norm(sv-sp)
        sv = sp
        I = I +1
    return R

def filtering_left_single_loop(Ts,R,s):
    size = len(Ts)
    for i in range(size,0,-1):
        T = ncon([Ts[s],R], [[-1,-2,1],[1,-3]])
        M = T.reshape([T.shape[0],T.shape[1]*T.shape[2]], order='F')
        _, R =  np.linalg.qr(M.T)
        R = R.T
        s = pbc_moving_left(s,size)

    R = R/np.max(abs(R))
    return R


def  projectors(Ts, EPS, MAX_I,chi1):

    projectorL=  [0 for x in range(len(Ts))]
    projectorR = [0 for x in range(len(Ts))]

    size = len(Ts)

    for s in range(size):
        sp = pbc_moving_left(s,size)
        L = filtering_right_loop(Ts,EPS, MAX_I, s)
        R = filtering_left_loop(Ts,EPS, MAX_I, sp)
        U, sv, V = np.linalg.svd(L@R,full_matrices=False)

        d =  sum(x>1E-12 for x in sv)
        chi = min(chi1,d)
        sv = sv[:chi]
        U = U[:,:chi]
        V= V[:chi,:]
        sv = (np.diag(sv**(-1/2)))

        projectorL[s] = sv@U.T@L
        projectorR[sp] = R@V.T@sv

    return projectorL,projectorR

def projecting(Ts,PL,PR):
    size = len(Ts)
    Ts_p =  [0 for x in range(len(Ts))]

    ORDER = [[1,-2,3],[-1,1],[3,-3]]
    for i in range(size):
        Ts_p[i] = ncon([Ts[i],  PL[i],  PR[i]], ORDER)
    return Ts_p

def filtering_periodic_mps( Ts, EPS, MAX_I,chi ):
    """
    Preconditioning for loop optimization, by
    L. Wang and F. Verstraete, Cluster update for tensor
    network states (2011), arXiv:1110.4362 [cond-mat.str-el]
    With this precondition, one can greatly enhance the convergence of loop optimization, although it should be noted that Loop-TNR still works without using this method.
    
    """
    PL,PR= projectors(Ts,EPS, MAX_I,chi )
    Ts_new = projecting(Ts,PL,PR)

    return Ts_new
