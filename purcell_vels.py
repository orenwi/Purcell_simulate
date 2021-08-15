import numpy as np
import math


def swimmer_model(state,t,phidotF,params):
    theta = state[2]
    phi = np.array([[state[3]],[state[4]]])
    G0, W = purcell_body_connection(phi,params)
    G = np.dot(Rotation_matrix_planar(theta),G0)
    phidot_n = np.array([[phidotF["phidot1F"](t)],[phidotF["phidot2F"](t)]])

    swimmer_vel = np.dot(G,phidot_n)
    state_dyn = np.concatenate((swimmer_vel, phidot_n), axis=0)
    state_dyn_l = state_dyn.reshape((5,)).tolist()
    return state_dyn_l

def Rotation_matrix_planar(theta):
    Rot = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
        ]) 
    return Rot


def resistance_tensor(alpha,l,ct):

    R = ct*l*np.array([[1+math.sin(alpha)**2, -math.sin(alpha)*math.cos(alpha),  0],
                [-math.sin(alpha)*math.cos(alpha),  1+math.cos(alpha)**2,      0],
                [ 0,                            0,               (l**2)/6]])
    return R


def purcell_body_connection(phi,params):
    phi1 = phi[0]
    phi2 = phi[1]
 
    l0 = params["l0"]
    l1 = params["l1"]
    l2 = params["l2"]
    ct = params["ct"]

    T0 = np.identity(3)
    T1 = np.array([[1, 0, -0.5*l1*math.sin(phi1)],
              [0, 1, -0.5*(l0+l1*math.cos(phi1))],
              [0, 0, 1]])
    T2 = np.array([[1, 0, -0.5*l2*math.sin(phi2)],
              [0, 1, 0.5*(l0+l2*math.cos(phi2))],
              [0, 0, 1]])

    E0 = np.zeros((3,2))
    E1 = np.array([[0.5*l1*math.sin(phi1), 0],
                [0.5*l1*math.cos(phi1), 0],
                [-1, 0]])
    E2 = np.array([[0, -0.5*l2*math.sin(phi2)],
                [0, 0.5*l2*math.cos(phi2)],
                [0, 1]])

    T = np.concatenate((T0, T1, T2), axis=0)
    E = np.concatenate((E0, E1, E2), axis=0)

    R0 = resistance_tensor(0,l0,ct)
    R1 = resistance_tensor(-phi1,l1,ct)
    R2 = resistance_tensor(phi2,l2,ct)
    R  = np.block([
            [R0, np.zeros((3,6))],
            [np.zeros((3,3)), R1, np.zeros((3,3))],
            [np.zeros((3,6)), R2]])

    Rbb = np.matmul(np.matmul(T.T,R),T)
    Rbu = np.matmul(np.matmul(T.T,R),E)
    Ruu = np.matmul(np.matmul(E.T,R),E)
    

    # detRbb = np.linalg.det(Rbb)

    G0 = -np.matmul(np.linalg.inv(Rbb), Rbu)
    W = Ruu+np.matmul(Rbu.T,G0)

    return G0, W


def Purcell_velocities(state,phidot,params):

    theta = state[2].copy()
    phi = state[0:2].copy()
    G0, W = purcell_body_connection(phi,params)
    G = np.dot(Rotation_matrix_planar(theta),G0)

    swimmer_vel = np.dot(G,phidot)

    return swimmer_vel