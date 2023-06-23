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

def  projectors(Ts, EPS, MAX_I ):
    projectorL = []
    projectorR = []
    for i in range(4):
        projectorL.append([])
        projectorR.append([])
    for s in range(4):
        sp = pbc_moving_left(s,4)
        L = filtering_right_loop(Ts,EPS, MAX_I, s)
        R = filtering_left_loop( Ts,EPS, MAX_I, sp)

        U, sv, V = np.linalg.svd(L@R,full_matrices=False)

        chi = sum(x>1E-12 for x in sv)
        sv = sv[:chi]
        U = U[:,:chi]
        V= V[:chi,:]
        sv = (np.diag(sv**(-1/2)))

        projectorL[s] = sv@U.T@L
        projectorR[sp] = R@V.T@sv

    return projectorL,projectorR

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
    for i in range(4):

        T = ncon([L,Ts[s]], [[-1,1],[1,-2,-3,-4]])
        _, L = np.linalg.qr(T.reshape([T.shape[0]*T.shape[1]*T.shape[2],T.shape[3]], order='F'))
        s = pbc_moving_right(s,4)

    L = L/np.max(abs(L))
    return L

def filtering_left_loop(Ts,eps, MAX_I, s ):
    SIZE = np.shape(Ts[s])
    R = np.eye(SIZE[3])
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
    for i in range(3,-1,-1):
        T = ncon([Ts[s],R], [[-1,-2,-3,1],[1,-4]])
        M = T.reshape([T.shape[0],T.shape[1]*T.shape[2]*T.shape[3]], order='F')
        _, R =  np.linalg.qr(M.T)
        R = R.T
        s = pbc_moving_left(s,4)
    R = R/np.max(abs(R))
    return R


def projecting(Ts,PL,PR):
    Ts[0] = ncon([Ts[0],  PL[0],  PR[2],  PL[2],  PR[0]], [[1,2,3,4],[-1,1],[2,-2],[-3,3],[4,-4]])
    Ts[1] = ncon([Ts[1],  PL[1],  PR[3],  PL[3],  PR[1]], [[1,2,3,4],[-1,1],[2,-2],[-3,3],[4,-4]])
    Ts[2] = ncon([Ts[2],  PL[2],  PR[0],  PL[0],  PR[2]], [[1,2,3,4],[-1,1],[2,-2],[-3,3],[4,-4]])
    Ts[3] = ncon([Ts[3],  PL[3],  PR[1],  PL[1],  PR[3]], [[1,2,3,4],[-1,1],[2,-2],[-3,3],[4,-4]])

    return Ts

def filtering( Ts, EPS, MAX_I ):
    """
    entanglement filtering procedures:
    this method filters out the Corner Double Line (CDL) tensor along square plaquette using the local gauges.

    Arguments:
        Ts: 4 of 4-index tensors
        EPS : Truncations threshold for "entanglement filtering".
        MAX_I : Number of iterations for "entanglement filtering".

    Output:
        Ts: Updated 4 of 4-index tensors
    """   

    PL,PR = projectors(Ts, EPS, MAX_I )
    Ts = projecting(Ts,PL,PR)
    return Ts
