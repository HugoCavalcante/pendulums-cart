# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 18:02:44 2024

@author: hugo


integrators_test.py
Various Integrators for differential equations
sdeint: Stochastic Differential Equations.
ddeint: Delay Differential Equations.
my_odeint: Ordinary Differential Equations (fixed order and step).



"""
#%% Imports
import numpy as np
from numpy.random import normal
import integrators as solve_edo
from matplotlib.pyplot import figure, plot, xlabel, ylabel, title, show, legend


#%% Testing Euler method
print("Testing Euler method:")
t = np.linspace(0, 10.0, num = 101)
h = t[1]-t[0]
def f(X,t):
    x, v = X[0], X[1]
    return np.array([v, -x])
X0 = np.array([1.0, 0.0])
X = solve_edo.Euler_method(f, X0, t)
x, v = X[:,0], X[:,1]
figure("Euler", figsize=(6,6))
title(f"Euler method with h = {h}")
plot(x, v, 'o-', markersize = 1.2)
theta = np.linspace(-np.pi, np.pi, num = 10001)
x_exact = np.sin(theta)
y_exact = np.cos(theta)
plot(x_exact, y_exact, color = 'gray')
xlabel("x")
ylabel("v")
show()

#%% Testing implicit Euler method
print("Testing implicit (backward) Euler method:")
t = np.linspace(0, 10.0, num = 101)
h = t[1]-t[0]
def f(X,t):
    x, v = X[0], X[1]
    return np.array([v, -x])
X0 = np.array([1.0, 0.0])
X = solve_edo.backward_Euler(f, X0, t)
x, v = X[:,0], X[:,1]
figure("Implicit Euler", figsize=(6,6))
title(f"Implicit Euler method with h = {h}")
plot(x, v, 'o-', markersize = 1.2)
theta = np.linspace(-np.pi, np.pi, num = 10001)
x_exact = np.sin(theta)
y_exact = np.cos(theta)
plot(x_exact, y_exact, color = 'gray')
xlabel("x")
ylabel("v")
show()

#%% Testing Euler-Cromer method
print("Testing Euler-Cromer (semi-implicit, symplectic) method:")
t = np.linspace(0, 10.0, num = 101)
h = t[1]-t[0]
def f(x, v, t):
    return v
def g(x, v,t):
    return -x


X0 = np.array([1.0, 0.0])
X = solve_edo.Euler_Cromer_method(f, g, X0[0], X0[1], t)
x, v = X[:,0], X[:,1]
figure("Euler-Cromer", figsize=(6,6))
title(f"Euler-Cromer (symplectic) method with h = {h}")
plot(x, v, 'o-', markersize = 1.2)
theta = np.linspace(-np.pi, np.pi, num = 10001)
x_exact = np.sin(theta)
y_exact = np.cos(theta)
plot(x_exact, y_exact, color = 'gray')
xlabel("x")
ylabel("v")
show()


#%% Testing all previous methods and displaing results in a single figure
### All methods for ODEs in a single figure
def f(X,t):
    x, v = X[0], X[1]
    return np.array([v, -x])
X0 = np.array([1.0, 0.0])
X = solve_edo.Euler_method(f, X0, t)
x_Euler, v_Euler = X[:,0], X[:,1]
X = solve_edo.backward_Euler(f, X0, t)
x_implicit, v_implicit = X[:,0], X[:,1]
def f(x, v, t):
    return v
def g(x, v,t):
    return -x
X0 = np.array([1.0, 0.0])
X = solve_edo.Euler_Cromer_method(f, g, X0[0], X0[1], t)
x_Euler_Cromer, v_Euler_Cromer = X[:,0], X[:,1]
figure("Energy", figsize=(6,6))
plot(x_Euler, v_Euler, 'o-', markersize = 1.2, label = "Euler")
plot(x_implicit, v_implicit, 'o-', markersize = 1.2, label = "backward Euler")
plot(x_Euler_Cromer, v_Euler_Cromer, 'o-', markersize = 1.2, label = "Euler-Cromer")
theta = np.linspace(-np.pi, np.pi, num = 10001)
x_exact = np.sin(theta)
y_exact = np.cos(theta)
plot(x_exact, y_exact, color = 'gray', label = "Exact")
xlabel("x")
ylabel("v")
legend()
show()

#%% Testing Ruth symplectic method of 4th order
print("Testing Ruth4 (symplectic 4th-order) method:")
tf = 500.0 
h = 0.1
N = int(tf/h)+1
#t = np.linspace(0, 100.0, num = 1001)
t = np.linspace(0, tf, num = N)
h = t[1]-t[0]
def f(x, v, t):
    return v
def g(x, v, t):
    #return -x
    return -np.sin(x)
def F(X, t):
    x, v = X[0], X[1]
    return np.array([f(x,v,t), g(x,v,t)])
#X0 = np.array([1.0, 0.0])
X0 = np.array([3.10, 0.042])
#E0 = (X0[0]**2 + X0[1]**2)/2
E0 = 1-np.cos(X0[0]) + (X0[1]**2)/2
X = solve_edo.Ruth4_symplectic(f, g, X0[0], X0[1], t)
x_Ruth4, v_Ruth4 = X[:,0], X[:,1]
#E_Ruth4 = (x_Ruth4**2 + v_Ruth4**2)/2
E_Ruth4 = 1-np.cos(x_Ruth4) + (v_Ruth4**2)/2
X = solve_edo.my_odeint(F, X0, t)
x_RK4, v_RK4 = X[:,0], X[:,1]
#E_RK4 = (x_RK4**2 + v_RK4**2)/2
E_RK4 = 1-np.cos(x_RK4) + (v_RK4**2)/2
figure("Ruth4", figsize=(6,6))
title(f"Ruth4 (symplectic) method with h = {h}")
plot(x_RK4, v_RK4, 'o-', markersize = 1.2, label = "RK4")
plot(x_Ruth4, v_Ruth4, 'o-', markersize = 1.2, label = "Ruth4")
legend()

#theta = np.linspace(-np.pi, np.pi, num = 10001)
# x_exact = np.sin(theta)
# y_exact = np.cos(theta)
# plot(x_exact, y_exact, color = 'gray')
#xlabel("x")
#ylabel("v")
xlabel(r"$\theta$")
ylabel(r"$\dot{\theta}$")

show()
figure("Energy_4th_order")
#plot(t, E_RK4-0.5, label = "RK4")
#plot(t, E_Ruth4-0.5, label = "Ruth4")
plot(t, E_RK4-E0, label = "RK4")
plot(t, E_Ruth4-E0, label = "Ruth4")

xlabel("time")
ylabel("E-E0")
legend()
show()

#%% Testando integrador simplético em sistema 6D (3 posições e 3 velocidades)
### Três pêndulos independentes
# def f(x, v, t):
#     return v
# def g(x, v, t):
#     #return -x
#     q1, q2, q3 = x
#     qp1, qp2, qp3 = v
#     return np.array([ 
#         -np.sin(q1),
#         -np.sqrt(2)*np.sin(q2),
#         -np.sqrt(3)*np.sin(q3)
#         ])


### Pêndulos no carrinho
tf = 3.0
h = 0.001
N = int(tf/h)+1
t = np.linspace(0, tf, num = N)
#h = t[1]-t[0]

m1, m2, M = 1.0, 1.0, 1.0
l1, l2 = 1, 1
gamma1, gamma2, gammac = 0, 0, 0

def f(x, v, t):
    q1, q2, xc, qp1, qp2, vc = x[0], x[1], x[2], v[0], v[1], v[2]
    #vcp = (m1*sin(q1)*(l1*qp1**2+g*cos(q1)) +m2*sin(q2)*(l2*qp2**2+g*cos(q2))+(gamma1/l1)*qp1*cos(q1)+(gamma2/l2)*qp2*cos(q2) -gammac*vc)/(M-m1*cos(q1)**2 -m2*cos(q2)**2)
    return np.array([
        qp1, 
        qp2, 
        vc
        ])

#IIE = 10
def g(x, v, t):
    q1, q2, xc, qp1, qp2, vc = x[0], x[1], x[2], v[0], v[1], v[2]
    vcp = (m1*np.sin(q1)*(l1*qp1**2+np.cos(q1)) +m2*np.sin(q2)*(l2*qp2**2+np.cos(q2))+(gamma1/l1)*qp1*np.cos(q1)+(gamma2/l2)*qp2*np.cos(q2) -gammac*vc)/(M-m1*np.cos(q1)**2 -m2*np.cos(q2)**2)
    # qp1_local = -(1/l1)*np.sin(q1) -vcp*np.cos(q1)/l1 - gamma1*qp1/(m1*l1**2)
    # qp2_local = -(1/l2)*np.sin(q2) -vcp*np.cos(q2)/l2 - gamma2*qp2/(m2*l2**2)
    # vc_local = vcp
    # for j in range(IIE):
    #     vcp = (m1*np.sin(q1)*(l1*qp1_local**2+np.cos(q1)) +m2*np.sin(q2)*(l2*qp2_local**2+np.cos(q2))+(gamma1/l1)*qp1_local*np.cos(q1)+(gamma2/l2)*qp2_local*np.cos(q2) -gammac*vc_local)/(M-m1*np.cos(q1)**2 -m2*np.cos(q2)**2)
    #     #vcp = (m1*np.sin(q1)*(l1*qp1_local**2+np.cos(q1)) +m2*np.sin(q2)*(l2*qp2_local**2+np.cos(q2))+(gamma1/l1)*qp1_local*np.cos(q1)+(gamma2/l2)*qp2_local*np.cos(q2) -gammac*vc)/(M-m1*np.cos(q1)**2 -m2*np.cos(q2)**2)
    #     qp1_local = -(1/l1)*np.sin(q1) -vcp*np.cos(q1)/l1 - gamma1*qp1_local/(m1*l1**2)
    #     qp2_local = -(1/l2)*np.sin(q2) -vcp*np.cos(q2)/l2 - gamma2*qp2_local/(m2*l2**2)
    #     vc_local = vcp
    # vcp = (m1*np.sin(q1)*(l1*qp1_local**2+np.cos(q1)) +m2*np.sin(q2)*(l2*qp2_local**2+np.cos(q2))+(gamma1/l1)*qp1_local*np.cos(q1)+(gamma2/l2)*qp2_local*np.cos(q2) -gammac*vc_local)/(M-m1*np.cos(q1)**2 -m2*np.cos(q2)**2)
    return np.array([
        -(1/l1)*np.sin(q1) -vcp*np.cos(q1)/l1 - gamma1*qp1/(m1*l1**2),
        -(1/l2)*np.sin(q2) -vcp*np.cos(q2)/l2 - gamma2*qp2/(m2*l2**2),
        vcp
        ])


#X0 = np.array([1.2, 1.1, 1.0, 0.0, 0.0, 0.0])
X0 = np.array([np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0])
#E0 = (X0[0]**2 + X0[1]**2)/2
#E0 = 1-np.cos(X0[0]) + (X0[1]**2)/2
X = solve_edo.Ruth4_symplectic(f, g, X0[0:3], X0[3:], t)
q1, q2, q3, qp1, qp2, qp3 = X[:,0], X[:,1], X[:,2], X[:,3], X[:,4], X[:,5]
#plot(t, q1, t, q2, t, q3)
plot(t, q1, label = "q1")
plot(t, q2, label = "q2") 
plot(t, q3, label = "q3")
legend()
xlabel("time")
ylabel("$q_{1,2,3}$")


