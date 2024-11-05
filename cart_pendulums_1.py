# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:18:48 2022

Solução numérica das Equações de movimento dos pêndulos no carrinho,
com pêndulos idênticos (mesma massa e comprimento). 
Solução em termos dos modos simétrico e assimétrico.

@author: Hugo Cavalcante
"""

from numpy import *
from matplotlib.pyplot import *
from scipy.integrate import odeint


# number of points in time series
#N = 10000
#time step
dt = 0.0001
#tf = (N-1)*dt
tf = 150
N = int(tf/dt)+1


gamma1 = 0.05 #5e-5 #0.001
gammac = 0.005 #5e-3 #0.01
beta = 2/3

# initial conditions
#X0 = array([1.5, 1.48801, 2.0, -2, 0, 0])
#X0 = array([0.0, 0.0, 0.0, 20.0, 0.5, 0])
X0 = array([0.0, 0.0, 0.0, 20.0, 0.5, 0])

def f(X,t):
    qa, qs, qc , va, vs, vc = X[0], X[1], X[2], X[3], X[4], X[5]
    vcp = (sin(qs)*(sin(qa)**2*cos(qs) + va**2*cos(qa)) + cos(qa)*(cos(qs)**2*sin(qa)+vs**2*sin(qs)) 
           +2*va*vs*sin(qa)*cos(qs) +gamma1*(vs*cos(qa)*cos(qs)-va*sin(qa)*sin(qs)) 
           -gammac*vc/2)/((1/beta+1)-(cos(qa)*cos(qs))**2 -(sin(qa)*sin(qs))**2)
    ### "4.37": movement equations for relative coordinates
    return array([
        va, 
        vs, 
        vc,
        vcp*sin(qa)*sin(qs) -sin(qa)*cos(qs) - gamma1*va,
        vcp*cos(qa)*cos(qs) -cos(qa)*sin(qs) - gamma1*vs,
        vcp
        ])
    

### eliminate transient
#t = linspace(0, 200.0, 20000)
#X = odeint(f, X0, t, atol = 1e-12)
#X0 = X[-1,:]

t = linspace(0.0, tf, N)
X = odeint(f, X0, t, atol = 1e-12)

qa, qs, qc = X[:,0], X[:,1], X[:,2]
q1 = qa+qs
q2 = qs-qa
qpa, qps, vc = X[:,3], X[:,4], X[:,5]
qp1 = qpa+qps
qp2 = qps-qpa


E1 = (1/beta+1)*vc**2 +qpa**2 +2*vc*qps*cos(qa)*cos(qs) + 2*vc*qpa*sin(qa)*sin(qs) -cos(qa)*cos(qs) +2
E2 = (1/beta+1)*vc**2 +0.5*(qp1**2+qp2**2) + vc*(qp1*cos(q1)+qp2*cos(q2)) -cos(q1)-cos(q2) +2
##plot(t, qa, label = '$q_a$')
##plot(t, qs, label = '$q_s$')
#plot(t, q1, label = '$q_1$')
#plot(t, q2, label = '$q_2$')
#plot(t, qc, label = '$q_c$')
plot(t[::10], vc[::10], label = '$v_c$')
#plot(t[::10], E1[::10], label = '$E_A$')
#plot(t[::10], E2[::10], label = '$E_B$')
xlabel('time')
ylabel('$q,v$')
legend()