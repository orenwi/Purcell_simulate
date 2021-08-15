from scipy.integrate import odeint
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from purcell_vels import *


params = {"l0" : 1,
          "l1" : 1,
          "l2" : 1,
          "ct": 1}


phiF = {"phi1F" : lambda x:  math.cos(x+math.pi/4),
        "phi2F" : lambda x:  math.sin(x+math.pi/4)}
phidotF = {"phidot1F" : lambda x:  -math.sin(x+math.pi/4),
        "phidot2F" : lambda x:  math.cos(x+math.pi/4)}

y0 = np.array([0,0,0,phiF["phi1F"](0),phiF["phi2F"](0)])
t = np.linspace(0,2*math.pi,400)
result = odeint(swimmer_model,y0,t,args=(phidotF,params))

# print(result)
fig1,ax1 = plt.subplots()
# ax1.plot(t,result[:,0],label='x(t)')
# ax1.plot(t,result[:,1],label='y(t)')
# ax1.plot(t,result[:,2],label='theta(t)')
# ax1.legend()
# ax1.set_xlabel('t')
# ax1.set_ylabel('Rp')

# fig2,ax2 = plt.subplots()
ax1.plot(result[:,3],result[:,4],label='phi')
ax1.axis('equal')
ax1.set_xlabel(r'$\phi_1$')#'\\textit{Velocity (\N{DEGREE SIGN}/sec)}'
ax1.set_ylabel(r'$\phi_2$')


fig2,ax2 = plt.subplots()
ax2.plot(t,result[:,1],label='x(t)')

# plt.show()
print('net X displacement is: ' + str(result[-1,0]))
print('net Y displacement is: ' + str(result[-1,1]))
print('net theta rotation is: ' + str(result[-1,2]))

