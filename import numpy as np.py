import numpy as np
import math


def resistance_tensor(alpha,l,ct):

    R = ct*l*np.array([[1+math.sin(alpha)**2, -math.sin(alpha)*math.cos(alpha),  0],
                [-math.sin(alpha)*math.cos(alpha),  1+math.cos(alpha)**2,      0],
                [ 0,                            0,               (l**2)/6]])
    return R


def purcel_body_velsM(phi,params):
    phi1 = phi["phi1"]
    phi2 = phi["phi2"]
    # phi1dot = phidot["phi1dot"]
    # phi2dot = phidot["phi2dot"]

    

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
    R1 = resistance_tensor(phi1,l1,ct)
    R2 = resistance_tensor(phi2,l2,ct)
    R  = np.block([
            [R0, np.zeros((3,6))],
            [np.zeros((3,3)), R1, np.zeros((3,3))],
            [np.zeros((3,6)), R2]])

    Rbb = np.matmul(np.matmul(T.T,R),T)
    Rbu = np.matmul(np.matmul(T.T,R),E)
      
    detRbb = np.linalg.det(Rbb)

    G = -np.matmul(np.linalg.inv(Rbb), Rbu)
    # Vel = np.dot(G,phidot)
    return G
          
    


phi = {"phi1" : 0,
       "phi2" : -math.pi/2}

# phidot = {"phi1dot" : 0,
    #    "phi2dot" : 0} 

phidot = np.array([[0],[0]],)
params = {"l0" : 1,
          "l1" : 1,
          "l2" : 1,
          "ct": 1}

G = purcel_body_velsM(phi,params)
vel = np.dot(G,phidot)
print(G)


# print("Velocities:"+ str(vel))






